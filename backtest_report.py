# backtest_report.py
# Backtest Statistics and Report Generation
# Calculates performance metrics from signal records
# Generates comprehensive backtest reports

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import logging
from datetime import datetime
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """Record of a single signal result"""
    timestamp: datetime
    symbol: str
    pattern: str
    direction: str  # BUY/SELL
    confidence: float
    entry_price: float
    stop_loss: float
    target_price: float
    rrr: float
    win_rate: float
    regime: str
    tier: str
    
    # Results (after signal is closed)
    status: str = "OPEN"  # OPEN, CLOSED_WIN, CLOSED_LOSS
    close_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    pnl_amount: Optional[float] = None
    duration_hours: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'symbol': self.symbol,
            'pattern': self.pattern,
            'direction': self.direction,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target_price': self.target_price,
            'rrr': self.rrr,
            'win_rate': self.win_rate,
            'regime': self.regime,
            'tier': self.tier,
            'status': self.status,
            'close_price': self.close_price,
            'pnl_pct': self.pnl_pct,
            'pnl_amount': self.pnl_amount,
            'duration_hours': self.duration_hours
        }


@dataclass
class BacktestMetrics:
    """Complete backtest performance metrics"""
    # Signal counts
    total_signals: int = 0
    signals_sent: int = 0  # MEDIUM+ tier only
    signals_open: int = 0
    signals_closed: int = 0
    
    # Win/Loss counts
    closed_wins: int = 0
    closed_losses: int = 0
    win_rate: float = 0.0
    
    # Financial metrics
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    
    # Advanced metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # RRR metrics
    avg_rrr: float = 0.0
    best_rrr: float = 0.0
    worst_rrr: float = 0.0
    
    # Tier breakdown
    premium_signals: int = 0
    high_signals: int = 0
    medium_signals: int = 0
    low_signals: int = 0
    rejected_signals: int = 0
    
    # Pattern performance
    pattern_accuracy: Dict[str, float] = field(default_factory=dict)
    regime_accuracy: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'total_signals': self.total_signals,
            'signals_sent': self.signals_sent,
            'signals_open': self.signals_open,
            'signals_closed': self.signals_closed,
            'closed_wins': self.closed_wins,
            'closed_losses': self.closed_losses,
            'win_rate': round(self.win_rate, 4),
            'total_pnl': round(self.total_pnl, 2),
            'total_pnl_pct': round(self.total_pnl_pct, 2),
            'profit_factor': round(self.profit_factor, 2),
            'max_drawdown': round(self.max_drawdown, 4),
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_losses': self.max_consecutive_losses,
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'calmar_ratio': round(self.calmar_ratio, 2),
            'avg_rrr': round(self.avg_rrr, 2),
            'best_rrr': round(self.best_rrr, 2),
            'worst_rrr': round(self.worst_rrr, 2),
            'tier_breakdown': {
                'premium': self.premium_signals,
                'high': self.high_signals,
                'medium': self.medium_signals,
                'low': self.low_signals,
                'rejected': self.rejected_signals
            },
            'pattern_accuracy': {k: round(v, 4) for k, v in self.pattern_accuracy.items()},
            'regime_accuracy': {k: round(v, 4) for k, v in self.regime_accuracy.items()}
        }


class BacktestReport:
    """Generate comprehensive backtest reports"""
    
    def __init__(self, signals: List[SignalRecord]):
        """Initialize report generator with signal records"""
        self.signals = signals
        self.logger = logging.getLogger(__name__)
        self.metrics = self._calculate_metrics()
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate all metrics from signals"""
        metrics = BacktestMetrics()
        metrics.total_signals = len(self.signals)
        
        # Filter closed signals for analysis
        closed_signals = [s for s in self.signals if s.status != "OPEN"]
        open_signals = [s for s in self.signals if s.status == "OPEN"]
        
        metrics.signals_open = len(open_signals)
        metrics.signals_closed = len(closed_signals)
        
        if not closed_signals:
            return metrics
        
        # Calculate win/loss metrics
        winning_signals = [s for s in closed_signals if s.status == "CLOSED_WIN"]
        losing_signals = [s for s in closed_signals if s.status == "CLOSED_LOSS"]
        
        metrics.closed_wins = len(winning_signals)
        metrics.closed_losses = len(losing_signals)
        metrics.win_rate = metrics.closed_wins / len(closed_signals) if closed_signals else 0
        
        # Calculate financial metrics
        metrics.total_pnl = sum(s.pnl_amount or 0 for s in closed_signals)
        metrics.total_pnl_pct = sum(s.pnl_pct or 0 for s in closed_signals)
        
        # Calculate profit factor
        gross_profit = sum(s.pnl_amount or 0 for s in winning_signals)
        gross_loss = abs(sum(s.pnl_amount or 0 for s in losing_signals))
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calculate RRR metrics
        rrr_values = [s.rrr for s in closed_signals]
        metrics.avg_rrr = np.mean(rrr_values) if rrr_values else 0
        metrics.best_rrr = max(rrr_values) if rrr_values else 0
        metrics.worst_rrr = min(rrr_values) if rrr_values else 0
        
        # Calculate advanced metrics
        pnl_values = np.array([s.pnl_pct or 0 for s in closed_signals])
        
        # Sharpe ratio
        if len(pnl_values) > 1 and np.std(pnl_values) > 0:
            metrics.sharpe_ratio = (np.mean(pnl_values) / np.std(pnl_values)) * np.sqrt(252)
        
        # Sortino ratio (downside deviation only)
        downside_returns = pnl_values[pnl_values < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            excess_return = np.mean(pnl_values)
            downside_dev = np.std(downside_returns)
            metrics.sortino_ratio = (excess_return / downside_dev) * np.sqrt(252)
        
        # Drawdown
        cumulative_pnl = np.cumsum(pnl_values)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - running_max) / np.abs(running_max) if np.any(running_max != 0) else np.zeros_like(cumulative_pnl)
        metrics.max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0
        
        # Calmar ratio
        if metrics.max_drawdown < 0:
            total_return = (np.sum(pnl_values) / 100)  # Simplified
            metrics.calmar_ratio = abs(total_return / metrics.max_drawdown)
        
        # Tier breakdown
        for signal in self.signals:
            if signal.tier == "PREMIUM":
                metrics.premium_signals += 1
            elif signal.tier == "HIGH":
                metrics.high_signals += 1
            elif signal.tier == "MEDIUM":
                metrics.medium_signals += 1
            elif signal.tier == "LOW":
                metrics.low_signals += 1
            else:
                metrics.rejected_signals += 1
        
        metrics.signals_sent = (metrics.premium_signals + metrics.high_signals + 
                               metrics.medium_signals)
        
        # Pattern accuracy
        metrics.pattern_accuracy = self._calculate_pattern_accuracy()
        metrics.regime_accuracy = self._calculate_regime_accuracy()
        
        # Consecutive wins/losses
        metrics.consecutive_wins = self._get_consecutive_count("CLOSED_WIN", closed_signals)
        metrics.consecutive_losses = self._get_consecutive_count("CLOSED_LOSS", closed_signals)
        metrics.max_consecutive_losses = self._get_max_consecutive_losses(closed_signals)
        
        return metrics
    
    def _calculate_pattern_accuracy(self) -> Dict[str, float]:
        """Calculate win rate for each pattern"""
        pattern_stats = {}
        
        for signal in self.signals:
            if signal.status == "OPEN":
                continue
            
            if signal.pattern not in pattern_stats:
                pattern_stats[signal.pattern] = {'wins': 0, 'total': 0}
            
            pattern_stats[signal.pattern]['total'] += 1
            if signal.status == "CLOSED_WIN":
                pattern_stats[signal.pattern]['wins'] += 1
        
        return {
            pattern: stats['wins'] / stats['total']
            for pattern, stats in pattern_stats.items()
            if stats['total'] > 0
        }
    
    def _calculate_regime_accuracy(self) -> Dict[str, float]:
        """Calculate win rate for each market regime"""
        regime_stats = {}
        
        for signal in self.signals:
            if signal.status == "OPEN":
                continue
            
            if signal.regime not in regime_stats:
                regime_stats[signal.regime] = {'wins': 0, 'total': 0}
            
            regime_stats[signal.regime]['total'] += 1
            if signal.status == "CLOSED_WIN":
                regime_stats[signal.regime]['wins'] += 1
        
        return {
            regime: stats['wins'] / stats['total']
            for regime, stats in regime_stats.items()
            if stats['total'] > 0
        }
    
    def _get_consecutive_count(self, status: str, closed_signals: List[SignalRecord]) -> int:
        """Get most recent consecutive count of a status"""
        count = 0
        for signal in reversed(closed_signals):
            if signal.status == status:
                count += 1
            else:
                break
        return count
    
    def _get_max_consecutive_losses(self, closed_signals: List[SignalRecord]) -> int:
        """Get maximum consecutive losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for signal in closed_signals:
            if signal.status == "CLOSED_LOSS":
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def get_metrics(self) -> BacktestMetrics:
        """Get calculated metrics"""
        return self.metrics
    
    def get_summary(self) -> Dict:
        """Get summary report"""
        return {
            'title': 'Backtest Summary Report',
            'generated_at': datetime.now().isoformat(),
            'total_signals': self.metrics.total_signals,
            'signals_sent': self.metrics.signals_sent,
            'closed_signals': self.metrics.signals_closed,
            'win_rate': f"{self.metrics.win_rate:.1%}",
            'profit_factor': f"{self.metrics.profit_factor:.2f}x",
            'sharpe_ratio': f"{self.metrics.sharpe_ratio:.2f}",
            'max_drawdown': f"{self.metrics.max_drawdown:.1%}",
            'total_pnl': f"₹{self.metrics.total_pnl:.2f}",
            'avg_rrr': f"{self.metrics.avg_rrr:.2f}:1"
        }
    
    def get_detailed_report(self) -> Dict:
        """Get detailed report with all metrics"""
        return {
            'summary': self.get_summary(),
            'metrics': self.metrics.to_dict(),
            'signals': [s.to_dict() for s in self.signals]
        }
    
    def export_report(self, filepath: str) -> None:
        """Export detailed report to JSON"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.get_detailed_report(), f, indent=2)
            self.logger.info(f"Backtest report exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export backtest report: {e}")
    
    def export_signals(self, filepath: str) -> None:
        """Export signals to JSON"""
        try:
            signals_data = [s.to_dict() for s in self.signals]
            with open(filepath, 'w') as f:
                json.dump(signals_data, f, indent=2)
            self.logger.info(f"Signals exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export signals: {e}")
    
    def print_summary(self) -> None:
        """Print summary to console"""
        print("\n" + "="*60)
        print("BACKTEST SUMMARY REPORT")
        print("="*60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*60)
        print(f"Total Signals:        {self.metrics.total_signals:>6}")
        print(f"Signals Sent:         {self.metrics.signals_sent:>6}")
        print(f"Closed Signals:       {self.metrics.signals_closed:>6}")
        print(f"Open Signals:         {self.metrics.signals_open:>6}")
        print("-"*60)
        print(f"Winning Signals:      {self.metrics.closed_wins:>6}")
        print(f"Losing Signals:       {self.metrics.closed_losses:>6}")
        print(f"Win Rate:             {self.metrics.win_rate:>5.1%}")
        print("-"*60)
        print(f"Total P&L:            ₹{self.metrics.total_pnl:>9,.2f}")
        print(f"Total P&L %:          {self.metrics.total_pnl_pct:>5.2f}%")
        print(f"Profit Factor:        {self.metrics.profit_factor:>6.2f}x")
        print("-"*60)
        print(f"Sharpe Ratio:         {self.metrics.sharpe_ratio:>6.2f}")
        print(f"Sortino Ratio:        {self.metrics.sortino_ratio:>6.2f}")
        print(f"Max Drawdown:         {self.metrics.max_drawdown:>5.1%}")
        print("-"*60)
        print(f"Avg RRR:              {self.metrics.avg_rrr:>6.2f}:1")
        print(f"Best RRR:             {self.metrics.best_rrr:>6.2f}:1")
        print(f"Worst RRR:            {self.metrics.worst_rrr:>6.2f}:1")
        print("-"*60)
        
        if self.metrics.pattern_accuracy:
            print("\nPATTERN ACCURACY:")
            for pattern, accuracy in sorted(self.metrics.pattern_accuracy.items(), 
                                          key=lambda x: x[1], reverse=True):
                print(f"  {pattern:<30} {accuracy:>5.1%}")
        
        if self.metrics.regime_accuracy:
            print("\nREGIME ACCURACY:")
            for regime, accuracy in sorted(self.metrics.regime_accuracy.items(), 
                                         key=lambda x: x[1], reverse=True):
                print(f"  {regime:<30} {accuracy:>5.1%}")
        
        print("="*60 + "\n")
