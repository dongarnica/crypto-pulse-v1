"""
Safe mathematical operations to prevent division by zero and other numerical errors.
"""
import numpy as np
import pandas as pd
from typing import Union, Any


def safe_div(numerator: Union[float, int, np.ndarray, pd.Series], 
             denominator: Union[float, int, np.ndarray, pd.Series], 
             fill_value: float = 0.0) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely divide numerator by denominator, replacing division by zero with fill_value.
    
    Args:
        numerator: The numerator value(s)
        denominator: The denominator value(s) 
        fill_value: Value to use when denominator is zero or invalid
        
    Returns:
        Result of safe division
    """
    if isinstance(denominator, (pd.Series, np.ndarray)):
        return np.where(
            (denominator != 0) & ~np.isnan(denominator) & ~np.isinf(denominator),
            numerator / denominator,
            fill_value
        )
    else:
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return fill_value
        return numerator / denominator


def safe_percentage(value: Union[float, int, np.ndarray, pd.Series], 
                   base: Union[float, int, np.ndarray, pd.Series], 
                   fill_value: float = 0.0) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely calculate percentage change: (value - base) / base * 100
    
    Args:
        value: Current value
        base: Base value for percentage calculation
        fill_value: Value to use when base is zero or invalid
        
    Returns:
        Percentage change
    """
    return safe_div(value - base, base, fill_value) * 100


def safe_ratio(numerator: Union[float, int, np.ndarray, pd.Series], 
               denominator: Union[float, int, np.ndarray, pd.Series], 
               fill_value: float = 1.0) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely calculate ratio, using fill_value when denominator is invalid.
    
    Args:
        numerator: The numerator value(s)
        denominator: The denominator value(s)
        fill_value: Value to use when denominator is zero or invalid (default 1.0 for ratios)
        
    Returns:
        Safe ratio calculation
    """
    return safe_div(numerator, denominator, fill_value)


def safe_log(value: Union[float, int, np.ndarray, pd.Series], 
             fill_value: float = 0.0) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely calculate natural logarithm, handling zero and negative values.
    
    Args:
        value: Value(s) to take log of
        fill_value: Value to use for zero or negative inputs
        
    Returns:
        Safe logarithm calculation
    """
    if isinstance(value, (pd.Series, np.ndarray)):
        return np.where(
            (value > 0) & ~np.isnan(value) & ~np.isinf(value),
            np.log(value),
            fill_value
        )
    else:
        if value <= 0 or np.isnan(value) or np.isinf(value):
            return fill_value
        return np.log(value)


def safe_sqrt(value: Union[float, int, np.ndarray, pd.Series], 
              fill_value: float = 0.0) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely calculate square root, handling negative values.
    
    Args:
        value: Value(s) to take square root of
        fill_value: Value to use for negative inputs
        
    Returns:
        Safe square root calculation
    """
    if isinstance(value, (pd.Series, np.ndarray)):
        return np.where(
            (value >= 0) & ~np.isnan(value) & ~np.isinf(value),
            np.sqrt(value),
            fill_value
        )
    else:
        if value < 0 or np.isnan(value) or np.isinf(value):
            return fill_value
        return np.sqrt(value)


def safe_risk_reward_ratio(stop_loss_distance: Union[float, int, np.ndarray, pd.Series],
                          take_profit_distance: Union[float, int, np.ndarray, pd.Series],
                          fill_value: float = 0.0) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely calculate risk-reward ratio: take_profit_distance / stop_loss_distance
    
    Args:
        stop_loss_distance: Distance to stop loss (risk amount)
        take_profit_distance: Distance to take profit (reward amount)
        fill_value: Value to use when stop_loss_distance is zero
        
    Returns:
        Safe risk-reward ratio
    """
    return safe_div(take_profit_distance, stop_loss_distance, fill_value)


def clean_numerical_data(data: Union[pd.DataFrame, pd.Series, np.ndarray], 
                        fill_method: str = 'forward') -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Clean numerical data by removing inf values and handling NaN values.
    
    Args:
        data: Data to clean
        fill_method: Method to handle NaN values ('forward', 'backward', 'zero', 'drop')
        
    Returns:
        Cleaned data
    """
    if isinstance(data, pd.DataFrame):
        # Replace inf/-inf with NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        
        if fill_method == 'forward':
            data = data.fillna(method='ffill').fillna(method='bfill')
        elif fill_method == 'backward':
            data = data.fillna(method='bfill').fillna(method='ffill')
        elif fill_method == 'zero':
            data = data.fillna(0)
        elif fill_method == 'drop':
            data = data.dropna()
            
    elif isinstance(data, pd.Series):
        # Replace inf/-inf with NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        
        if fill_method == 'forward':
            data = data.fillna(method='ffill').fillna(method='bfill')
        elif fill_method == 'backward':
            data = data.fillna(method='bfill').fillna(method='ffill')
        elif fill_method == 'zero':
            data = data.fillna(0)
        elif fill_method == 'drop':
            data = data.dropna()
            
    elif isinstance(data, np.ndarray):
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    return data


def validate_numerical_inputs(*args, operation_name: str = "calculation") -> bool:
    """
    Validate that numerical inputs are valid for calculations.
    
    Args:
        *args: Numerical values to validate
        operation_name: Name of operation for error messages
        
    Returns:
        True if all inputs are valid, False otherwise
    """
    for i, arg in enumerate(args):
        if isinstance(arg, (int, float)):
            if np.isnan(arg) or np.isinf(arg):
                print(f"Warning: Invalid input {i+1} for {operation_name}: {arg}")
                return False
        elif isinstance(arg, (pd.Series, np.ndarray)):
            if np.any(np.isnan(arg)) or np.any(np.isinf(arg)):
                print(f"Warning: Invalid values found in input {i+1} for {operation_name}")
                return False
    
    return True


# Legacy compatibility - maintain existing safe_div function signatures
def safe_division(a, b, fill_val=0):
    """Legacy compatibility wrapper for safe_div"""
    return safe_div(a, b, fill_val)
