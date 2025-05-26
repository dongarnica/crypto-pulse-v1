"""
Display formatting utilities for crypto trading application.
Provides consistent formatting for trade status, recommendations, and other data.
"""

from typing import Dict, List, Any
import time
from datetime import datetime
from enum import Enum
import textwrap

class DisplayColors:
    """ASCII color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


class TradeStatus(Enum):
    """Trade status types with associated colors."""
    PENDING = (DisplayColors.YELLOW, "⏳")
    SUCCESS = (DisplayColors.GREEN, "✅")
    PARTIAL = (DisplayColors.BLUE, "⚠️") 
    FAILED = (DisplayColors.RED, "❌")
    CANCELED = (DisplayColors.MAGENTA, "🚫")
    TIMEOUT = (DisplayColors.RED, "⏱️")


class TradeFormatter:
    """Formats trade execution information for display."""
    
    @staticmethod
    def format_trade_status(status: str) -> str:
        """Format a trade status with color and emoji."""
        status_lower = status.lower()
        
        if "success" in status_lower or "executed" in status_lower:
            trade_status = TradeStatus.SUCCESS
        elif "partial" in status_lower:
            trade_status = TradeStatus.PARTIAL
        elif "fail" in status_lower or "error" in status_lower:
            trade_status = TradeStatus.FAILED
        elif "cancel" in status_lower:
            trade_status = TradeStatus.CANCELED
        elif "timeout" in status_lower:
            trade_status = TradeStatus.TIMEOUT
        else:
            trade_status = TradeStatus.PENDING
            
        color, emoji = trade_status.value
        return f"{color}{emoji} {status.title()}{DisplayColors.RESET}"
    
    @staticmethod
    def format_trade_execution(trade_data: Dict) -> str:
        """Format trade execution information with details."""
        # Extract basic trade info
        status = trade_data.get('status', 'Unknown')
        action = trade_data.get('action', 'Unknown')
        
        # Format the status with color
        formatted_status = TradeFormatter.format_trade_status(status)
        
        # Extract trade details
        data = trade_data.get('data', {})
        if not data:
            return f"{formatted_status} - {action.replace('_', ' ').title()}"
        
        # Format basic trade information
        symbol = data.get('symbol', 'Unknown')
        side = data.get('side', '').title()
        entry_price = data.get('entry_price', 0)
        quantity = data.get('quantity', 0)
        
        # Additional details for closed positions
        pnl = data.get('pnl', None)
        pnl_pct = data.get('pnl_pct', None)
        
        # Build result string
        result = f"{formatted_status} - {action.replace('_', ' ').title()}"
        result += f" | {symbol} {side} {abs(quantity):.6f} @ ${entry_price:.2f}"
        
        # Add P&L information if available (for closed positions)
        if pnl is not None:
            pnl_color = DisplayColors.GREEN if pnl >= 0 else DisplayColors.RED
            result += f" | P&L: {pnl_color}${pnl:.2f} ({pnl_pct:+.2f}%){DisplayColors.RESET}"
            
        return result
    
    @staticmethod
    def format_trade_list(trades: List[Dict], max_items: int = 5) -> str:
        """Format a list of trades for display."""
        if not trades:
            return "No trade executions to display"
            
        result = []
        for i, trade in enumerate(trades[:max_items], 1):
            result.append(f"{i}. {TradeFormatter.format_trade_execution(trade)}")
            
        if len(trades) > max_items:
            result.append(f"... and {len(trades) - max_items} more trades")
            
        return "\n".join(result)


class RecommendationFormatter:
    """Formats trading recommendations for display."""
    
    PRIORITY_COLORS = {
        'urgent': DisplayColors.BG_RED + DisplayColors.WHITE,
        'high': DisplayColors.RED,
        'medium': DisplayColors.YELLOW,
        'low': DisplayColors.BLUE,
    }
    
    ACTION_EMOJIS = {
        'hold': '⏹️',
        'close': '⛔',
        'reduce': '⬇️',
        'add': '⬆️',
        'rebalance': '♻️',
        'set_stop_loss': '🛑',
        'set_take_profit': '🎯',
    }
    
    @staticmethod
    def format_recommendation(rec: Dict) -> str:
        """Format a single recommendation with color and details."""
        priority = rec.get('priority', '').lower()
        action = rec.get('action', '').lower()
        symbol = rec.get('symbol', 'Unknown')
        reason = rec.get('reason', '')
        confidence = rec.get('confidence', 0) * 100
        
        # Get color based on priority
        color = RecommendationFormatter.PRIORITY_COLORS.get(priority, DisplayColors.WHITE)
        emoji = RecommendationFormatter.ACTION_EMOJIS.get(action, '🔹')
        
        # Add price information if available
        current_price = rec.get('current_price', 0)
        target_price = rec.get('target_price')
        price_info = f" @ ${current_price:.2f}"
        
        if target_price:
            price_change = ((target_price / current_price) - 1) * 100
            price_info += f" → ${target_price:.2f} ({price_change:+.2f}%)"
        
        # Format the recommendation
        header = f"{color}{priority.upper()}{DisplayColors.RESET} {emoji} {action.upper()} {symbol}{price_info}"
        details = f"   {reason} (Confidence: {confidence:.1f}%)"
        
        return f"{header}\n{details}"
    
    @staticmethod
    def format_recommendation_summary(recommendations: Dict) -> str:
        """Format a summary of all recommendations by priority."""
        if not recommendations:
            return "No recommendations available"
            
        urgent = recommendations.get('urgent', [])
        high = recommendations.get('high', [])
        medium = recommendations.get('medium', [])
        low = recommendations.get('low', [])
        
        total = len(urgent) + len(high) + len(medium) + len(low)
        
        if total == 0:
            return "No active recommendations"
            
        urgent_color = RecommendationFormatter.PRIORITY_COLORS.get('urgent')
        high_color = RecommendationFormatter.PRIORITY_COLORS.get('high')
        medium_color = RecommendationFormatter.PRIORITY_COLORS.get('medium')
        low_color = RecommendationFormatter.PRIORITY_COLORS.get('low')
        
        summary = f"TOTAL RECOMMENDATIONS: {total}\n"
        summary += f"  {urgent_color}URGENT{DisplayColors.RESET}: {len(urgent)}"
        summary += f" | {high_color}HIGH{DisplayColors.RESET}: {len(high)}"
        summary += f" | {medium_color}MEDIUM{DisplayColors.RESET}: {len(medium)}"
        summary += f" | {low_color}LOW{DisplayColors.RESET}: {len(low)}"
        
        return summary
    
    @staticmethod
    def format_recommendations_by_priority(recommendations: Dict, max_per_level: int = 3) -> str:
        """Format recommendations grouped by priority level."""
        if not recommendations:
            return "No recommendations available"
            
        result = [RecommendationFormatter.format_recommendation_summary(recommendations)]
        
        # Format each priority level
        for priority in ['urgent', 'high', 'medium', 'low']:
            recs = recommendations.get(priority, [])
            if not recs:
                continue
                
            # Get color for this priority
            color = RecommendationFormatter.PRIORITY_COLORS.get(priority, DisplayColors.WHITE)
            result.append(f"\n{color}{priority.upper()} RECOMMENDATIONS{DisplayColors.RESET}")
            
            # Format each recommendation in this priority level
            displayed = 0
            for rec in recs[:max_per_level]:
                result.append(RecommendationFormatter.format_recommendation(rec))
                displayed += 1
                
            # Show how many more are hidden
            if len(recs) > max_per_level:
                result.append(f"   ... and {len(recs) - max_per_level} more {priority} recommendations")
                
        return "\n".join(result)


def format_currency(amount: float) -> str:
    """Format a currency value."""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format a percentage value with +/- sign."""
    return f"{value:+.2f}%"


def get_timestamp() -> str:
    """Get a formatted timestamp for current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_progress_bar(current: int, total: int, width: int = 20, fill_char: str = "█", empty_char: str = "░") -> str:
    """Create a text-based progress bar."""
    progress = min(1.0, current / max(1, total))
    filled = int(width * progress)
    bar = fill_char * filled + empty_char * (width - filled)
    percent = int(progress * 100)
    return f"{bar} {percent}%"


def boxed_message(message: str, width: int = 80, title: str = None) -> str:
    """Create a boxed message with optional title."""
    lines = []
    lines.append("┌" + "─" * (width - 2) + "┐")
    
    if title:
        title_padding = (width - len(title) - 4) // 2
        lines.append("│" + " " * title_padding + title + " " * (width - title_padding - len(title) - 2) + "│")
        lines.append("│" + "─" * (width - 2) + "│")
    
    # Wrap message to fit in box
    wrapped_lines = textwrap.wrap(message, width=width-4)
    for line in wrapped_lines:
        padding = width - len(line) - 2
        lines.append("│ " + line + " " * padding + "│")
    
    lines.append("└" + "─" * (width - 2) + "┘")
    return "\n".join(lines)
