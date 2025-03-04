import numpy as np
from typing import Tuple, List, Optional, Dict, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

class ConformalPredictor:
    def __init__(self, significance: float = 0.1):
        self.significance = significance
        self.calibration_scores = None
        
    def _process_input(self, batch_input: Union[Dict[str, torch.Tensor], torch.Tensor], device: str) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Helper method to process input consistently"""
        if isinstance(batch_input, dict):
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch_input.items()}
        return batch_input.to(device) if isinstance(batch_input, torch.Tensor) else batch_input
        
    def calibrate(self, model: nn.Module, calibration_loader: DataLoader, device: str) -> None:
        """Calibrate the conformal predictor using a calibration set"""
        model.eval()
        scores = []
        
        with torch.no_grad():
            for inputs, labels in calibration_loader:
                # Move all tensors to device before forward pass
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(device)
                if isinstance(inputs, dict):
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in inputs.items()}
                elif isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                
                logits = model(inputs)
                softmax_probs = torch.softmax(logits, dim=1)
                batch_scores = 1 - torch.max(softmax_probs, dim=1)[0]
                scores.extend(batch_scores.cpu().numpy())
        
        self.calibration_scores = np.array(scores)
        print(f"Calibrated with {len(scores)} samples")
        
    def get_prediction_sets(self, logits: torch.Tensor) -> Tuple[List[List[int]], np.ndarray]:
        """Get prediction sets with guaranteed coverage"""
        if self.calibration_scores is None:
            raise ValueError("Calibration has not been performed")
            
        # Calculate quantile - ensure it's in range [0,1]
        n = self.calibration_scores.shape[0]
        calibration_level = np.ceil((n + 1) * (1 - self.significance)) / n
        # Clamp to valid range
        calibration_level = min(max(calibration_level, 0.0), 1.0)
        threshold = np.quantile(self.calibration_scores, calibration_level)
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Calculate prediction sets
        pred_sets = []
        scores = []
        
        for prob in probs:
            # Get indices where 1 - prob <= threshold
            pred_set = torch.where(1 - prob <= threshold)[0].cpu().numpy()
            
            # If prediction set is empty, include at least the highest probability class
            if len(pred_set) == 0:
                pred_set = torch.tensor([torch.argmax(prob)]).cpu().numpy()
            
            pred_sets.append(pred_set.tolist())
            scores.append(1 - torch.max(prob).item())
            
        return pred_sets, np.array(scores)
    
    def save_calibration(self, filepath: str) -> None:
        """Save calibration scores to a file"""
        if self.calibration_scores is None:
            raise ValueError("No calibration scores to save")
        
        # Use pathlib for safe path handling
        filepath_path = Path(filepath)
        # Ensure parent directory exists
        filepath_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(filepath_path, self.calibration_scores)
        
    def load_calibration(self, filepath: str) -> None:
        """Load calibration scores from a file"""
        # Use pathlib for safe path resolution and validation
        filepath_path = Path(filepath).resolve()
        
        if not filepath_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {filepath_path}")
        
        self.calibration_scores = np.load(filepath_path)
        print(f"Loaded calibration with {len(self.calibration_scores)} samples")

def split_calibration_data(dataset: Dataset, calibration_ratio: float = 0.2) -> Tuple[Dataset, Dataset]:
    """Split dataset into training and calibration sets"""
    indices = list(range(len(dataset)))
    train_idx, calib_idx = train_test_split(indices, test_size=calibration_ratio, random_state=42)
    
    class SubsetDataset(Dataset):
        def __init__(self, dataset: Dataset, indices: List[int]):
            self.dataset = dataset
            self.indices = indices
            
        def __len__(self):
            return len(self.indices)
            
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]
    
    return SubsetDataset(dataset, train_idx), SubsetDataset(dataset, calib_idx)