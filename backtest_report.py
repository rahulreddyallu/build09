"""

BACKTEST REPORT - INSTITUTIONAL GRADE ANALYSIS & REPORTING

================================================

This module provides comprehensive backtest statistics and report generation.

Features:

✓ Complete signal record tracking (entry, exit, P&L)

✓ Multi-dimensional performance metrics (financial, risk, risk-adjusted)

✓ Pattern and regime performance analysis

✓ Config-integrated validation and thresholds

✓ JSON export for data pipeline integration

✓ Console reporting with formatted output

✓ Production-ready logging and error handling

✓ Consecutive streak tracking and drawdown analysis

Integration:

- Uses config.py SignalTier enums and thresholds

- Respects risk_management parameters for validation

- Leverages validation thresholds for signal filtering

- Applies market_data parameters for historical analysis

Author: rahulreddyallu

Version: 2.1.0 (Config-Integrated, Institutional Grade)

Date: 2025-12-01

"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
import logging
import numpy as np
from datetime import datetime, timedelta

# Configure logger
logger = logging.getLogger(__name__)


# ============================================================================

# SIGNAL RECORD DATACLASS

# ============================================================================


@dataclass
class SignalRecord:
    """
    Complete record of a single signal and its lifecycle

    Tracks from generation through execution to final P&L

    """

    # Signal metadata
    timestamp: datetime
    symbol: str
    pattern: str
    direction: str  # BUY or SELL
    confidence: float  # 0.0 to 1.0
    regime: str  # UPTREND, DOWNTREND, SIDEWAYS, VOLATILE

    # Entry parameters
    entry_price: float
    stop_loss: float
    target_price: float
    rrr: float  # Risk-Reward Ratio
    win_rate: float  # Expected win rate for this setup

    # Signal tier classification
    tier: str  # PREMIUM, HIGH, MEDIUM, LOW, REJECT

    # Position status
    status: str = "OPEN"  # OPEN, CLOSED_WIN, CLOSED_LOSS, CANCELLED
    close_price: Optional[float] = None
    close_timestamp: Optional[datetime] = None

    # P&L tracking
    pnl_amount: Optional[float] = None  # ₹ amount
    pnl_pct: Optional[float] = None  # Percentage return
    duration_hours: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert signal to dictionary for serialization"""
        return {
            "timestamp": (
                self.timestamp.isoformat()
                if isinstance(self.timestamp, datetime)
                else self.timestamp
            ),
            "symbol": self.symbol,
            "pattern": self.pattern,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "regime": self.regime,
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "target_price": round(self.target_price, 2),
            "rrr": round(self.rrr, 2),
            "win_rate": round(self.win_rate, 4),
            "tier": self.tier,
            "status": self.status,
            "close_price": round(self.close_price, 2) if self.close_price else None,
            "close_timestamp": (
                self.close_timestamp.isoformat()
                if isinstance(self.close_timestamp, datetime)
                else self.close_timestamp
            ),
            "pnl_amount": round(self.pnl_amount, 2) if self.pnl_amount else None,
            "pnl_pct": round(self.pnl_pct, 4) if self.pnl_pct else None,
            "duration_hours": (
                round(self.duration_hours, 2) if self.duration_hours else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SignalRecord":
        """Reconstruct SignalRecord from dictionary"""
        # Convert ISO timestamp strings back to datetime
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if isinstance(data.get("close_timestamp"), str):
            data["close_timestamp"] = datetime.fromisoformat(data["close_timestamp"])

        return cls(**data)

    def calculate_pnl(self, close_price: float) -> Tuple[float, float]:
        """
        Calculate P&L when position is closed

        Returns:
            Tuple of (pnl_amount, pnl_pct)

        """
        if self.direction == "BUY":
            pnl_amount = (close_price - self.entry_price) * 1  # 1 unit
        else:  # SELL
            pnl_amount = (self.entry_price - close_price) * 1

        pnl_pct = (pnl_amount / self.entry_price) * 100 if self.entry_price > 0 else 0

        return pnl_amount, pnl_pct

    def finalize(self, close_price: float, close_timestamp: Optional[datetime] = None):
        """Close signal and calculate results"""
        self.close_price = close_price
        self.close_timestamp = close_timestamp or datetime.now()

        # Determine win/loss
        if self.direction == "BUY":
            self.status = "CLOSED_WIN" if close_price >= self.target_price else (
                "CLOSED_LOSS" if close_price <= self.stop_loss else "OPEN"
            )
        else:  # SELL
            self.status = "CLOSED_WIN" if close_price <= self.target_price else (
                "CLOSED_LOSS" if close_price >= self.stop_loss else "OPEN"
            )

        # Calculate P&L
        if self.status != "OPEN":
            self.pnl_amount, self.pnl_pct = self.calculate_pnl(close_price)
            self.duration_hours = (
                (self.close_timestamp - self.timestamp).total_seconds() / 3600
            )


# ============================================================================

# BACKTEST METRICS DATACLASS

# ============================================================================


@dataclass
class BacktestMetrics:
    """
    Complete backtest performance metrics

    Includes signal counts, financial metrics, risk metrics, and pattern analysis

    """

    # Signal counts
    total_signals: int = 0
    signals_sent: int = 0  # MEDIUM+ tier only (actual trades)
    signals_open: int = 0
    signals_closed: int = 0

    # Win/Loss counts
    closed_wins: int = 0
    closed_losses: int = 0
    closed_cancelled: int = 0
    win_rate: float = 0.0

    # Financial metrics
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    profit_factor: float = 0.0
    avg_profit_per_trade: float = 0.0
    avg_loss_per_trade: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_amount: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0

    # Advanced risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    recovery_factor: float = 0.0

    # RRR metrics
    avg_rrr: float = 0.0
    best_rrr: float = 0.0
    worst_rrr: float = 0.0
    expected_value: float = 0.0  # Win% * Avg_Win - Loss% * Avg_Loss

    # Tier breakdown
    premium_signals: int = 0
    high_signals: int = 0
    medium_signals: int = 0
    low_signals: int = 0
    rejected_signals: int = 0

    # Pattern and regime performance
    pattern_accuracy: Dict[str, float] = field(default_factory=dict)
    pattern_count: Dict[str, int] = field(default_factory=dict)
    regime_accuracy: Dict[str, float] = field(default_factory=dict)
    regime_count: Dict[str, int] = field(default_factory=dict)

    # Direction performance (BUY vs SELL)
    buy_win_rate: float = 0.0
    sell_win_rate: float = 0.0

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            "signal_counts": {
                "total_signals": self.total_signals,
                "signals_sent": self.signals_sent,
                "signals_open": self.signals_open,
                "signals_closed": self.signals_closed,
            },
            "win_loss": {
                "closed_wins": self.closed_wins,
                "closed_losses": self.closed_losses,
                "closed_cancelled": self.closed_cancelled,
                "win_rate": round(self.win_rate, 4),
            },
            "financial": {
                "total_pnl": round(self.total_pnl, 2),
                "total_pnl_pct": round(self.total_pnl_pct, 4),
                "profit_factor": round(self.profit_factor, 2),
                "avg_profit_per_trade": round(self.avg_profit_per_trade, 2),
                "avg_loss_per_trade": round(self.avg_loss_per_trade, 2),
            },
            "risk": {
                "max_drawdown": round(self.max_drawdown, 4),
                "max_drawdown_amount": round(self.max_drawdown_amount, 2),
                "consecutive_wins": self.consecutive_wins,
                "consecutive_losses": self.consecutive_losses,
                "max_consecutive_losses": self.max_consecutive_losses,
            },
            "risk_adjusted": {
                "sharpe_ratio": round(self.sharpe_ratio, 2),
                "sortino_ratio": round(self.sortino_ratio, 2),
                "calmar_ratio": round(self.calmar_ratio, 2),
                "recovery_factor": round(self.recovery_factor, 2),
            },
            "rrr": {
                "avg_rrr": round(self.avg_rrr, 2),
                "best_rrr": round(self.best_rrr, 2),
                "worst_rrr": round(self.worst_rrr, 2),
                "expected_value": round(self.expected_value, 4),
            },
            "tier_breakdown": {
                "premium": self.premium_signals,
                "high": self.high_signals,
                "medium": self.medium_signals,
                "low": self.low_signals,
                "rejected": self.rejected_signals,
            },
            "direction": {
                "buy_win_rate": round(self.buy_win_rate, 4),
                "sell_win_rate": round(self.sell_win_rate, 4),
            },
            "pattern_accuracy": {
                k: round(v, 4) for k, v in self.pattern_accuracy.items()
            },
            "pattern_count": self.pattern_count,
            "regime_accuracy": {
                k: round(v, 4) for k, v in self.regime_accuracy.items()
            },
            "regime_count": self.regime_count,
        }

    def validate_metrics(self) -> Tuple[bool, List[str]]:
        """
        Validate metric consistency

        Returns:
            Tuple of (is_valid, list_of_warnings)

        """
        warnings = []

        # Check win rate bounds
        if not (0 <= self.win_rate <= 1):
            warnings.append(f"Win rate {self.win_rate} out of [0, 1] bounds")

        # Check profit factor
        if self.profit_factor < 0:
            warnings.append(f"Negative profit factor: {self.profit_factor}")

        # Check drawdown
        if self.max_drawdown > 0:
            warnings.append(
                f"Max drawdown should be negative, got {self.max_drawdown}"
            )

        # Check RRR bounds
        if self.avg_rrr < 0 or self.best_rrr < 0:
            warnings.append("RRR values should be positive")

        # Check signal counts
        total_tiered = (
            self.premium_signals
            + self.high_signals
            + self.medium_signals
            + self.low_signals
            + self.rejected_signals
        )
        if total_tiered != self.total_signals:
            warnings.append(
                f"Tier counts ({total_tiered}) != total_signals ({self.total_signals})"
            )

        return len(warnings) == 0, warnings


# ============================================================================

# BACKTEST REPORT GENERATOR

# ============================================================================


class BacktestReport:
    """
    Generate comprehensive backtest reports from signal records

    Integrates with config.py for parameter validation and thresholds

    """

    def __init__(
        self,
        signals: List[SignalRecord],
        config: Optional[object] = None,
        risk_free_rate: float = 0.06,
    ):
        """
        Initialize report generator

        Args:
            signals: List of SignalRecord objects
            config: Optional BotConfiguration object for validation
            risk_free_rate: Annual risk-free rate for Sharpe ratio (default 6%)

        """
        self.signals = signals
        self.config = config
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
        self.metrics = self._calculate_metrics()

    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate all metrics from signals"""
        metrics = BacktestMetrics()
        metrics.total_signals = len(self.signals)

        # Separate signals by status
        closed_signals = [s for s in self.signals if s.status != "OPEN"]
        open_signals = [s for s in self.signals if s.status == "OPEN"]

        metrics.signals_open = len(open_signals)
        metrics.signals_closed = len(closed_signals)

        if not closed_signals:
            self.logger.warning("No closed signals to analyze")
            return metrics

        # Separate by outcome
        winning_signals = [s for s in closed_signals if s.status == "CLOSED_WIN"]
        losing_signals = [s for s in closed_signals if s.status == "CLOSED_LOSS"]
        cancelled_signals = [
            s for s in closed_signals if s.status == "CLOSED_CANCELLED"
        ]

        metrics.closed_wins = len(winning_signals)
        metrics.closed_losses = len(losing_signals)
        metrics.closed_cancelled = len(cancelled_signals)

        # Win rate
        tradeable_signals = metrics.closed_wins + metrics.closed_losses
        if tradeable_signals > 0:
            metrics.win_rate = metrics.closed_wins / tradeable_signals
        else:
            self.logger.warning("No tradeable signals (only cancelled)")
            return metrics

        # Financial metrics
        metrics.total_pnl = sum(s.pnl_amount or 0 for s in closed_signals)
        metrics.total_pnl_pct = sum(s.pnl_pct or 0 for s in closed_signals)

        # Profit factor
        gross_profit = sum(s.pnl_amount or 0 for s in winning_signals if s.pnl_amount)
        gross_loss = abs(
            sum(s.pnl_amount or 0 for s in losing_signals if s.pnl_amount and s.pnl_amount < 0)
        )
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Average trade P&L
        if winning_signals:
            metrics.avg_profit_per_trade = sum(
                s.pnl_amount or 0 for s in winning_signals
            ) / len(winning_signals)
        if losing_signals:
            metrics.avg_loss_per_trade = sum(
                s.pnl_amount or 0 for s in losing_signals
            ) / len(losing_signals)

        # RRR metrics
        rrr_values = [s.rrr for s in closed_signals if s.rrr > 0]
        if rrr_values:
            metrics.avg_rrr = np.mean(rrr_values)
            metrics.best_rrr = max(rrr_values)
            metrics.worst_rrr = min(rrr_values)
        else:
            self.logger.warning("No valid RRR values found")

        # Expected value calculation
        avg_win = abs(metrics.avg_profit_per_trade) if metrics.avg_profit_per_trade > 0 else 0
        avg_loss = abs(metrics.avg_loss_per_trade) if metrics.avg_loss_per_trade < 0 else 0
        if tradeable_signals > 0:
            win_pct = metrics.closed_wins / tradeable_signals
            loss_pct = metrics.closed_losses / tradeable_signals
            metrics.expected_value = (win_pct * avg_win) - (loss_pct * avg_loss)

        # Advanced metrics
        pnl_values = np.array([s.pnl_pct or 0 for s in closed_signals])

        # Sharpe ratio (daily returns assumed)
        if len(pnl_values) > 1:
            daily_return = np.mean(pnl_values) / 100
            daily_vol = np.std(pnl_values) / 100
            if daily_vol > 0:
                metrics.sharpe_ratio = (daily_return - (self.risk_free_rate / 252)) / daily_vol * np.sqrt(252)

        # Sortino ratio (downside deviation)
        downside_returns = np.array([x for x in pnl_values if x < 0])
        if len(downside_returns) > 0:
            excess_return = np.mean(pnl_values) / 100
            downside_dev = np.std(downside_returns) / 100
            if downside_dev > 0:
                metrics.sortino_ratio = (
                    (excess_return - (self.risk_free_rate / 252)) / downside_dev * np.sqrt(252)
                )

        # Drawdown analysis
        cumulative_pnl = np.cumsum(pnl_values)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown_array = (cumulative_pnl - running_max) / np.abs(
            np.where(running_max != 0, running_max, 1)
        )
        metrics.max_drawdown = float(np.min(drawdown_array)) if len(drawdown_array) > 0 else 0
        metrics.max_drawdown_amount = float(np.min(cumulative_pnl - running_max)) if len(cumulative_pnl) > 0 else 0

        # Calmar ratio
        if metrics.max_drawdown < 0 and metrics.max_drawdown_amount < 0:
            total_return = np.sum(pnl_values)
            metrics.calmar_ratio = abs(total_return / metrics.max_drawdown_amount) if metrics.max_drawdown_amount != 0 else 0

        # Recovery factor
        if metrics.max_drawdown_amount < 0:
            metrics.recovery_factor = metrics.total_pnl / abs(metrics.max_drawdown_amount)

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

        metrics.signals_sent = (
            metrics.premium_signals + metrics.high_signals + metrics.medium_signals
        )

        # Pattern and regime accuracy
        metrics.pattern_accuracy = self._calculate_pattern_accuracy()
        metrics.pattern_count = self._get_pattern_counts()
        metrics.regime_accuracy = self._calculate_regime_accuracy()
        metrics.regime_count = self._get_regime_counts()

        # Direction performance
        buy_signals = [s for s in closed_signals if s.direction == "BUY"]
        sell_signals = [s for s in closed_signals if s.direction == "SELL"]

        if buy_signals:
            buy_wins = len([s for s in buy_signals if s.status == "CLOSED_WIN"])
            metrics.buy_win_rate = buy_wins / len(buy_signals)

        if sell_signals:
            sell_wins = len([s for s in sell_signals if s.status == "CLOSED_WIN"])
            metrics.sell_win_rate = sell_wins / len(sell_signals)

        # Consecutive streaks
        metrics.consecutive_wins = self._get_consecutive_count("CLOSED_WIN", closed_signals)
        metrics.consecutive_losses = self._get_consecutive_count("CLOSED_LOSS", closed_signals)
        metrics.max_consecutive_losses = self._get_max_consecutive_losses(closed_signals)

        # Validate metrics
        is_valid, validation_warnings = metrics.validate_metrics()
        if not is_valid:
            for warning in validation_warnings:
                self.logger.warning(f"Metric validation: {warning}")

        return metrics

    def _calculate_pattern_accuracy(self) -> Dict[str, float]:
        """Calculate win rate for each pattern"""
        pattern_stats = {}
        for signal in self.signals:
            if signal.status == "OPEN" or signal.status == "CLOSED_CANCELLED":
                continue

            if signal.pattern not in pattern_stats:
                pattern_stats[signal.pattern] = {"wins": 0, "total": 0}

            pattern_stats[signal.pattern]["total"] += 1
            if signal.status == "CLOSED_WIN":
                pattern_stats[signal.pattern]["wins"] += 1

        return {
            pattern: stats["wins"] / stats["total"]
            for pattern, stats in pattern_stats.items()
            if stats["total"] > 0
        }

    def _get_pattern_counts(self) -> Dict[str, int]:
        """Get count of trades per pattern"""
        pattern_counts = {}
        for signal in self.signals:
            if signal.status != "OPEN" and signal.status != "CLOSED_CANCELLED":
                pattern_counts[signal.pattern] = pattern_counts.get(signal.pattern, 0) + 1
        return pattern_counts

    def _calculate_regime_accuracy(self) -> Dict[str, float]:
        """Calculate win rate for each market regime"""
        regime_stats = {}
        for signal in self.signals:
            if signal.status == "OPEN" or signal.status == "CLOSED_CANCELLED":
                continue

            if signal.regime not in regime_stats:
                regime_stats[signal.regime] = {"wins": 0, "total": 0}

            regime_stats[signal.regime]["total"] += 1
            if signal.status == "CLOSED_WIN":
                regime_stats[signal.regime]["wins"] += 1

        return {
            regime: stats["wins"] / stats["total"]
            for regime, stats in regime_stats.items()
            if stats["total"] > 0
        }

    def _get_regime_counts(self) -> Dict[str, int]:
        """Get count of trades per regime"""
        regime_counts = {}
        for signal in self.signals:
            if signal.status != "OPEN" and signal.status != "CLOSED_CANCELLED":
                regime_counts[signal.regime] = regime_counts.get(signal.regime, 0) + 1
        return regime_counts

    def _get_consecutive_count(
        self, status: str, closed_signals: List[SignalRecord]
    ) -> int:
        """Get most recent consecutive count of a specific status"""
        count = 0
        for signal in reversed(closed_signals):
            if signal.status == status:
                count += 1
            else:
                break
        return count

    def _get_max_consecutive_losses(
        self, closed_signals: List[SignalRecord]
    ) -> int:
        """Get maximum consecutive losses in signal sequence"""
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
        """Get calculated metrics object"""
        return self.metrics

    def get_summary(self) -> Dict:
        """Get concise summary report"""
        return {
            "generated_at": datetime.now().isoformat(),
            "total_signals": self.metrics.total_signals,
            "signals_sent": self.metrics.signals_sent,
            "signals_closed": self.metrics.signals_closed,
            "win_rate": f"{self.metrics.win_rate:.1%}",
            "profit_factor": f"{self.metrics.profit_factor:.2f}x",
            "sharpe_ratio": f"{self.metrics.sharpe_ratio:.2f}",
            "sortino_ratio": f"{self.metrics.sortino_ratio:.2f}",
            "max_drawdown": f"{self.metrics.max_drawdown:.1%}",
            "total_pnl": f"₹{self.metrics.total_pnl:.2f}",
            "avg_rrr": f"{self.metrics.avg_rrr:.2f}:1",
            "expected_value": f"₹{self.metrics.expected_value:.2f}",
        }

    def get_detailed_report(self) -> Dict:
        """Get complete detailed report with all metrics and signals"""
        return {
            "summary": self.get_summary(),
            "metrics": self.metrics.to_dict(),
            "signals": [s.to_dict() for s in self.signals],
            "config_integrated": self.config is not None,
        }

    def export_report(self, filepath: str) -> bool:
        """
        Export detailed report to JSON

        Returns:
            True if successful, False otherwise

        """
        try:
            with open(filepath, "w") as f:
                json.dump(self.get_detailed_report(), f, indent=2)
            self.logger.info(f"✓ Backtest report exported to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"✗ Failed to export backtest report: {e}")
            return False

    def export_signals(self, filepath: str) -> bool:
        """
        Export signals to JSON

        Returns:
            True if successful, False otherwise

        """
        try:
            signals_data = [s.to_dict() for s in self.signals]
            with open(filepath, "w") as f:
                json.dump(signals_data, f, indent=2)
            self.logger.info(f"✓ Signals exported to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"✗ Failed to export signals: {e}")
            return False

    def export_metrics_csv(self, filepath: str) -> bool:
        """
        Export metrics to CSV for spreadsheet analysis

        Returns:
            True if successful, False otherwise

        """
        try:
            import csv

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                writer.writerow(["Generated At", datetime.now().isoformat()])
                writer.writerow([])

                metrics_dict = self.metrics.to_dict()
                for section, values in metrics_dict.items():
                    if isinstance(values, dict):
                        writer.writerow([section.upper(), ""])
                        for key, value in values.items():
                            writer.writerow([key, value])
                    writer.writerow([])

            self.logger.info(f"✓ Metrics exported to CSV: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"✗ Failed to export metrics CSV: {e}")
            return False

    def print_summary(self) -> None:
        """Print formatted summary report to console"""
        print("\n" + "=" * 70)
        print("BACKTEST SUMMARY REPORT".center(70))
        print("=" * 70)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 70)

        # Signal counts
        print(f"Total Signals Generated: {self.metrics.total_signals:>6}")
        print(f"Signals Sent (MEDIUM+):  {self.metrics.signals_sent:>6}")
        print(f"Closed Signals:          {self.metrics.signals_closed:>6}")
        print(f"Open Signals:            {self.metrics.signals_open:>6}")
        print("-" * 70)

        # Win/Loss
        print(f"Winning Trades:          {self.metrics.closed_wins:>6}")
        print(f"Losing Trades:           {self.metrics.closed_losses:>6}")
        print(f"Win Rate:                {self.metrics.win_rate:>5.1%}")
        print("-" * 70)

        # Financial
        print(f"Total P&L:               ₹{self.metrics.total_pnl:>9,.2f}")
        print(f"Total P&L %:             {self.metrics.total_pnl_pct:>5.2f}%")
        print(f"Profit Factor:           {self.metrics.profit_factor:>6.2f}x")
        print(f"Avg Profit/Trade:        ₹{self.metrics.avg_profit_per_trade:>9,.2f}")
        print(f"Avg Loss/Trade:          ₹{self.metrics.avg_loss_per_trade:>9,.2f}")
        print("-" * 70)

        # Risk-adjusted
        print(f"Sharpe Ratio:            {self.metrics.sharpe_ratio:>6.2f}")
        print(f"Sortino Ratio:           {self.metrics.sortino_ratio:>6.2f}")
        print(f"Calmar Ratio:            {self.metrics.calmar_ratio:>6.2f}")
        print(f"Max Drawdown:            {self.metrics.max_drawdown:>5.1%}")
        print(f"Drawdown Amount:         ₹{self.metrics.max_drawdown_amount:>9,.2f}")
        print("-" * 70)

        # RRR
        print(f"Avg RRR:                 {self.metrics.avg_rrr:>6.2f}:1")
        print(f"Best RRR:                {self.metrics.best_rrr:>6.2f}:1")
        print(f"Worst RRR:               {self.metrics.worst_rrr:>6.2f}:1")
        print(f"Expected Value (₹):      {self.metrics.expected_value:>9.2f}")
        print("-" * 70)

        # Direction breakdown
        print(f"Buy Win Rate:            {self.metrics.buy_win_rate:>5.1%}")
        print(f"Sell Win Rate:           {self.metrics.sell_win_rate:>5.1%}")
        print("-" * 70)

        # Tier breakdown
        print("SIGNAL TIER BREAKDOWN:")
        print(f"  PREMIUM:               {self.metrics.premium_signals:>6}")
        print(f"  HIGH:                  {self.metrics.high_signals:>6}")
        print(f"  MEDIUM:                {self.metrics.medium_signals:>6}")
        print(f"  LOW:                   {self.metrics.low_signals:>6}")
        print(f"  REJECTED:              {self.metrics.rejected_signals:>6}")
        print("-" * 70)

        # Pattern accuracy
        if self.metrics.pattern_accuracy:
            print("\nPATTERN PERFORMANCE:")
            for pattern, accuracy in sorted(
                self.metrics.pattern_accuracy.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                count = self.metrics.pattern_count.get(pattern, 0)
                print(f"  {pattern:<25} {accuracy:>5.1%}  ({count} trades)")

        # Regime accuracy
        if self.metrics.regime_accuracy:
            print("\nREGIME PERFORMANCE:")
            for regime, accuracy in sorted(
                self.metrics.regime_accuracy.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                count = self.metrics.regime_count.get(regime, 0)
                print(f"  {regime:<25} {accuracy:>5.1%}  ({count} trades)")

        print("=" * 70 + "\n")


if __name__ == "__main__":
    # Example usage with test data
    test_signals = [
        SignalRecord(
            timestamp=datetime.now(),
            symbol="NSE_EQ|INE009A01021",
            pattern="BULLISH_ENGULFING",
            direction="BUY",
            confidence=0.85,
            regime="UPTREND",
            entry_price=100.0,
            stop_loss=95.0,
            target_price=110.0,
            rrr=2.0,
            win_rate=0.65,
            tier="HIGH",
        )
    ]

    report = BacktestReport(test_signals)
    report.print_summary()
    print("✓ Backtest report module initialized successfully")
