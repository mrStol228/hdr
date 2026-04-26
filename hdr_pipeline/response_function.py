from __future__ import annotations
import math
import logging
from typing import List

class ResponseFunction:
    """Response function for HDR processing.
    
    Given a set of data Xi and Yi, this class estimates a relation between X and Y
    using linear least squares. Used to modify pixels of images taken at brighter 
    or darker exposure levels to estimate what the pixel should be at the "base" exposure.
    Estimates as y = parameter_A * x + parameter_B.
    """
    
    def __init__(self, parameter_a: float = 1.0, parameter_b: float = 0.0):
        self.parameter_a = parameter_a
        self.parameter_b = parameter_b
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def create_identity():
        """Create identity response function (y = x)."""
        return ResponseFunction(1.0, 0.0)
    
    @classmethod
    def create_from_samples(cls, x_samples: List[float], y_samples: List[float], 
                           weights: List[float], func_id: int = 0):
        """Computes the response function using linear least squares.
        
        Args:
            x_samples: List of Xi samples. Must be at least 4 samples.
            y_samples: List of Yi samples. Must be same length as x_samples.
            weights: List of weights. Must be same length as x_samples.
            func_id: Function ID for debugging.
            
        Returns:
            ResponseFunction instance.
            
        Raises:
            ValueError: If input arrays have different lengths or insufficient samples.
        """
        logger = logging.getLogger(__name__)
        
        if len(x_samples) != len(y_samples):
            logger.error("unequal number of samples")
            raise ValueError("unequal number of samples")
        
        if len(x_samples) != len(weights):
            logger.error("unequal number of samples")
            raise ValueError("unequal number of samples")
        
        if len(x_samples) <= 3:
            logger.error("not enough samples")
            raise ValueError("not enough samples")
        
                           
        sum_wx = sum(w * x for w, x in zip(weights, x_samples))
        sum_wx2 = sum(w * x * x for w, x in zip(weights, x_samples))
        sum_wxy = sum(w * x * y for w, x, y in zip(weights, x_samples, y_samples))
        sum_wy = sum(w * y for w, y in zip(weights, y_samples))
        sum_w = sum(weights)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"sum_wx = {sum_wx}")
            logger.debug(f"sum_wx2 = {sum_wx2}")
            logger.debug(f"sum_wxy = {sum_wxy}")
            logger.debug(f"sum_wy = {sum_wy}")
            logger.debug(f"sum_w = {sum_w}")
        
                              
                                             
                                                
            
                                                                 
                                                                        
                                                                              
        
        a_numer = sum_wy * sum_wx - sum_w * sum_wxy
        a_denom = sum_wx * sum_wx - sum_w * sum_wx2
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"A_numer = {a_numer}")
            logger.debug(f"A_denom = {a_denom}")
        
        done = False
        parameter_a = 1.0
        parameter_b = 0.0
        
        if abs(a_denom) >= 1e-5:
            parameter_a = a_numer / a_denom
            parameter_b = (sum_wy - parameter_a * sum_wx) / sum_w
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"parameter_A = {parameter_a}")
                logger.debug(f"parameter_B = {parameter_b}")
            
                                                                                 
            if parameter_a >= 1e-5 and parameter_b >= 1e-5:
                done = True
            else:
                if parameter_a < 1e-5:
                    logger.error(f"parameter A too small or negative: {parameter_a}")
                if parameter_b < 1e-5:
                    logger.error(f"parameter B too small or negative: {parameter_b}")
        else:
            logger.error("denom too small")
        
        if not done:
            logger.error("falling back to linear Y = AX")
                           
            numer = sum(w * x * y for w, x, y in zip(weights, x_samples, y_samples))
            denom = sum(w * x * x for w, x in zip(weights, x_samples))
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"numer = {numer}")
                logger.debug(f"denom = {denom}")
            
            if denom >= 1e-5:
                parameter_a = numer / denom
                                                                 
                if parameter_a < 1e-5:
                    logger.error(f"parameter A too small or negative: {parameter_a}")
                    parameter_a = 1e-5
            else:
                logger.error("denom too small")
                parameter_a = 1.0
            
            parameter_b = 0.0
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Final parameter_A = {parameter_a}")
            logger.debug(f"Final parameter_B = {parameter_b}")
        
        return cls(parameter_a, parameter_b)
    
    def apply(self, value: float) -> float:
        """Apply the response function to a value."""
        return self.parameter_a * value + self.parameter_b
    
    def __repr__(self):
        return f"ResponseFunction(A={self.parameter_a:.6f}, B={self.parameter_b:.6f})"