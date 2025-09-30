"""
Training curriculum for progressive learning.
"""


class TrainingCurriculum:
    """
    Manages training curriculum - progressive adjustment of
    loss weights and other hyperparameters.
    """
    
    def __init__(self):
        self.stage = 0
        
    def get_loss_weights(self, epoch):
        """
        Get loss weights for current epoch.
        
        Strategy:
        - Stage 1 (0-5 epochs): Focus on task performance
        - Stage 2 (6-15 epochs): Introduce sparsity
        - Stage 3 (16+ epochs): Balance all objectives
        
        Args:
            epoch: Current epoch number
            
        Returns:
            dict: Loss weights {alpha, beta, gamma}
        """
        if epoch < 5:
            # Stage 1: Task accuracy first
            return {
                'alpha': 1.0,  # Task loss
                'beta': 0.01,  # Sparsity loss (minimal)
                'gamma': 0.01  # Diversity loss (minimal)
            }
        elif epoch < 15:
            # Stage 2: Add sparsity objective
            return {
                'alpha': 1.0,
                'beta': 0.1,   # Increase sparsity weight
                'gamma': 0.05  # Encourage diversity
            }
        else:
            # Stage 3: Full multi-objective
            return {
                'alpha': 1.0,
                'beta': 0.2,   # Strong sparsity penalty
                'gamma': 0.1   # Strong diversity reward
            }
    
    def get_encoding_temperature(self, epoch, max_epochs=20):
        """
        Get temperature for encoding selection.
        
        Anneals from soft (high temp) to hard (low temp) selection.
        
        Args:
            epoch: Current epoch
            max_epochs: Total epochs for annealing
            
        Returns:
            float: Temperature value
        """
        # Linear annealing from 1.0 to 0.1
        progress = min(epoch / max_epochs, 1.0)
        temp = 1.0 - 0.9 * progress
        return max(temp, 0.1)  # Clamp to minimum 0.1
    
    def get_time_steps(self, epoch):
        """
        Get number of time steps T for current epoch.
        
        Can progressively increase temporal resolution.
        
        Args:
            epoch: Current epoch
            
        Returns:
            int: Number of time steps
        """
        if epoch < 10:
            return 4  # Start with T=4
        elif epoch < 20:
            return 6  # Increase to T=6
        else:
            return 8  # Final resolution T=8