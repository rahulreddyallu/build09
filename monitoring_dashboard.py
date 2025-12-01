"""
MONITORING DASHBOARD - LIVE MONITORING & ADHOC VALIDATION

==========================================================

This module provides:

✓ Real-time signal monitoring dashboard
✓ Adhoc manual signal validation interface
✓ Live performance statistics tracking
✓ Signal history and analytics
✓ Terminal-based interactive UI

Features:

- Display live stock prices and detected patterns
- Manual pattern validation with custom thresholds
- Real-time performance metrics
- Signal tier distribution
- Historical win rate tracking
- Market regime indicator
- Trade entry/exit management

Author: rahulreddyallu

Version: 4.0.0 (Institutional Grade)

Date: 2025-11-30
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import pandas as pd

from config import BotConfiguration, SignalTier
from market_analyzer import MarketAnalyzer, MarketRegime
from signal_validator import SignalValidator


@dataclass
class SignalRecord:
    """Record of a generated signal"""

    timestamp: datetime
    symbol: str
    direction: str  # BUY or SELL
    pattern: str
    tier: str
    confidence: int
    entry_price: float
    stop_loss: float
    target_price: float
    rrr: float
    win_rate: float
    status: str  # OPEN, CLOSED_WIN, CLOSED_LOSS, CLOSED_BREAK_EVEN
    close_price: Optional[float] = None
    close_timestamp: Optional[datetime] = None
    pnl_pct: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Daily/session performance statistics"""

    timestamp: datetime = field(default_factory=datetime.now)
    signals_generated: int = 0
    signals_by_tier: Dict[str, int] = field(default_factory=lambda: {
        'PREMIUM': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0
    })
    signals_sent: int = 0
    signals_open: int = 0
    signals_closed: int = 0

    # Performance metrics
    closed_wins: int = 0
    closed_losses: int = 0
    closed_breakeven: int = 0
    win_rate: float = 0.0
    avg_rrr_promised: float = 0.0
    avg_rrr_achieved: float = 0.0
    profit_factor: float = 0.0
    total_pnl_pct: float = 0.0

    # Risk tracking
    max_consecutive_losses: int = 0
    current_consecutive_losses: int = 0
    max_daily_drawdown: float = 0.0

    # Best/Worst patterns
    best_pattern: str = "N/A"
    best_pattern_winrate: float = 0.0
    worst_pattern: str = "N/A"
    worst_pattern_winrate: float = 0.0


class MonitoringDashboard:
    """
    Live monitoring dashboard for signals and performance

    Provides:
    - Real-time signal tracking
    - Performance statistics
    - Signal history analytics
    - Visual terminal interface
    """

    def __init__(
        self,
        config: BotConfiguration,
        analyzer: MarketAnalyzer,
        validator: SignalValidator,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize dashboard

        Args:
            config: BotConfiguration instance
            analyzer: MarketAnalyzer instance
            validator: SignalValidator instance
            logger: Optional logger
        """
        self.config = config
        self.analyzer = analyzer
        self.validator = validator
        self.logger = logger or logging.getLogger(__name__)

        # Signal history tracking
        self.signal_history: List[SignalRecord] = []
        self.performance_metrics = PerformanceMetrics()
        # Pattern accuracy tracking
        self.pattern_winrates: Dict[str, Dict[str, Any]] = {}

        # Load history if exists
        self._load_history()

    def _load_history(self):
        """Load signal history from file"""
        history_file = Path("signals_history.json")
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    # Parse history (simplified)
                    self.logger.info(f"✓ Loaded {len(data.get('signals', []))} historical signals")
                # Parsing omitted for brevity but can be implemented to populate self.signal_history
            except Exception as e:
                self.logger.warning(f"Could not load history: {str(e)}")

    def display_dashboard(self, stocks_data: List[Dict[str, Any]]) -> None:
        """
        Display live monitoring dashboard

        Args:
            stocks_data: List of analyzed stock data
        """
        self._clear_screen()
        self._print_header()
        self._print_current_signals(stocks_data)
        self._print_performance_stats()
        self._print_open_positions()
        self._print_footer()

    def _clear_screen(self):
        """Clear terminal screen"""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')

    def _print_header(self) -> None:
        """Print dashboard header"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 15 + "STOCK SIGNALLING BOT - LIVE DASHBOARD" + " " * 26 + "║")
        print("║" + " " * 20 + timestamp + " " * 33 + "║")
        print("╚" + "═" * 78 + "╝")
        print()

    def _print_current_signals(self, stocks_data: List[Dict[str, Any]]) -> None:
        """Print currently detected signals"""
        print("┌─ CURRENT SIGNALS " + "─" * 61 + "┐")

        signal_count = 0
        for stock in stocks_data:
            if not stock.get('valid'):
                continue

            patterns = stock.get('patterns', [])
            symbol = stock.get('symbol', 'UNKNOWN')
            current_price = stock.get('price', 0)
            regime = stock.get('market_regime', MarketRegime.RANGE).value

            for pattern in patterns:
                if signal_count >= 5:  # Show max 5 signals
                    print("│ ... and more │")
                    break
                emoji = "🟢" if pattern.pattern_type == "BULLISH" else "🔴"
                direction = "BUY" if pattern.pattern_type == "BULLISH" else "SELL"
                print(
                    f"│ {emoji} {symbol:8} | {direction:4} | {pattern.pattern_name:20} | "
                    f"Confidence: {pattern.confidence_score}/5 | Regime: {regime:15} │"
                )
                signal_count += 1

        if signal_count == 0:
            print("│ No patterns detected at this time │")
        print("└" + "─" * 78 + "┘")
        print()

    def _print_performance_stats(self) -> None:
        """Print performance statistics"""
        metrics = self.performance_metrics
        print("┌─ TODAY'S PERFORMANCE " + "─" * 57 + "┐")
        print(
            f"│ Signals Generated: {metrics.signals_generated:3} | "
            f"Sent: {metrics.signals_sent:3} | "
            f"Open: {metrics.signals_open:3} | "
            f"Closed: {metrics.signals_closed:3} │"
        )
        print("│ │")
        print(
            f"│ Win Rate: {metrics.win_rate*100:5.1f}% | "
            f"Wins: {metrics.closed_wins:2} | "
            f"Losses: {metrics.closed_losses:2} | "
            f"Profit Factor: {metrics.profit_factor:.2f}x │"
        )
        print(
            f"│ Avg RRR Promised: {metrics.avg_rrr_promised:5.2f}:1 | "
            f"Achieved: {metrics.avg_rrr_achieved:5.2f}:1 | "
            f"Total P&L: {metrics.total_pnl_pct:+6.2f}% │"
        )
        print("│ │")
        print(
            f"│ Best Pattern: {metrics.best_pattern:15} ({metrics.best_pattern_winrate*100:5.1f}%) | "
            f"Worst: {metrics.worst_pattern:15} ({metrics.worst_pattern_winrate*100:5.1f}%) │"
        )
        print("└" + "─" * 78 + "┘")
        print()

    def _print_open_positions(self) -> None:
        """Print open signal positions"""
        print("┌─ OPEN POSITIONS " + "─" * 62 + "┐")
        open_signals = [s for s in self.signal_history if s.status == "OPEN"]
        if not open_signals:
            print("│ No open positions │")
        else:
            for signal in open_signals[-5:]:  # Show last 5 open
                emoji = "🟢" if signal.direction == "BUY" else "🔴"
                time_open = signal.timestamp.strftime("%H:%M")
                pnl = "N/A"  # Would calculate based on current price
                print(
                    f"│ {emoji} {signal.symbol:8} | {signal.direction:4} | "
                    f"Entry: {signal.entry_price:8.2f} | "
                    f"Target: {signal.target_price:8.2f} | "
                    f"P&L: {pnl:>6} | Opened: {time_open} │"
                )
        print("└" + "─" * 78 + "┘")
        print()

    def _print_footer(self) -> None:
        """Print dashboard footer"""
        print("Commands: [v] Validate Signal | [h] History | [s] Stats | [q] Quit")


class AdhocSignalValidator:
    """
    Manual signal validation interface

    Allows manual pattern input and validation with custom thresholds
    """

    def __init__(
        self,
        config: BotConfiguration,
        analyzer: MarketAnalyzer,
        validator: SignalValidator,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize adhoc validator

        Args:
            config: BotConfiguration instance
            analyzer: MarketAnalyzer instance
            validator: SignalValidator instance
            logger: Optional logger
        """
        self.config = config
        self.analyzer = analyzer
        self.validator = validator
        self.logger = logger or logging.getLogger(__name__)

    def validate_manual_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        pattern_name: str,
        signal_direction: str,
        custom_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Manually validate a signal with optional custom thresholds

        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol
            pattern_name: Name of pattern to validate
            signal_direction: "BUY" or "SELL"
            custom_thresholds: Optional custom parameter overrides

        Returns:
            Detailed validation breakdown
        """
        self.logger.info(f"Adhoc validation requested: {symbol} {signal_direction} ({pattern_name})")

        # Run complete analysis
        analysis = self.analyzer.analyze_stock(df, symbol)
        if not analysis.get('valid'):
            return {
                'valid': False,
                'reason': analysis.get('reason', 'Analysis failed')
            }

        # Validate signal
        result = self.validator.validate_signal(
            df=df,
            symbol=symbol,
            signal_direction=signal_direction,
            pattern_name=pattern_name,
            current_price=df.iloc[-1]['Close']
        )

        # Format breakdown
        breakdown = {
            'symbol': symbol,
            'pattern': pattern_name,
            'direction': signal_direction,
            'timestamp': datetime.now().isoformat(),
            'validation_passed': result.validation_passed,
            'signal_tier': result.signal_tier.name,
            'confidence_score': result.confidence_score,

            # Component scores
            'pattern_score': result.pattern_score,
            'indicator_score': result.indicator_score,
            'context_score': result.context_score,
            'risk_score': result.risk_score,

            # Details
            'patterns_detected': result.patterns_detected,
            'supporting_indicators': result.supporting_indicators,
            'opposing_indicators': result.opposing_indicators,

            # Risk metrics
            'entry_price': result.risk_validation.entry_price if result.risk_validation else 0,
            'stop_loss': result.risk_validation.stop_loss if result.risk_validation else 0,
            'target_price': result.risk_validation.target_price if result.risk_validation else 0,
            'rrr': result.risk_validation.rrr if result.risk_validation else 0,

            'historical_win_rate': result.historical_win_rate,

            # Market context
            'market_regime': analysis.get('market_regime', 'UNKNOWN').value,
            'rsi_value': analysis.get('indicators', {}).rsi if analysis.get('indicators') else 0,
            'macd_signal': analysis.get('indicators', {}).macd_signal_dir if analysis.get('indicators') else 'N/A',

            # Rejection reason if failed
            'rejection_reason': result.rejection_reason if not result.validation_passed else None
        }

        self.logger.info(f"Validation complete: {result.signal_tier.name} ({result.confidence_score}/10)")

        return breakdown

    def display_validation_result(self, breakdown: Dict[str, Any]) -> None:
        """
        Display formatted validation result

        Args:
            breakdown: Validation breakdown dictionary
        """
        if not breakdown.get('validation_passed'):
            print("\n❌ SIGNAL VALIDATION FAILED")
            print(f"Reason: {breakdown.get('rejection_reason', 'Unknown')}")
            return

        symbol = breakdown['symbol']
        direction = breakdown['direction']
        tier = breakdown['signal_tier']
        confidence = breakdown['confidence_score']

        # Header
        emoji = "🟢" if direction == "BUY" else "🔴"
        print(f"\n{emoji} SIGNAL VALIDATED - {tier} TIER")
        print(f"Symbol: {symbol} | Direction: {direction} | Confidence: {confidence}/10")
        print()

        # Scoring breakdown
        print("Scoring Breakdown:")
        print(f" Pattern Score: {breakdown['pattern_score']}/3")
        print(f" Indicator Score: {breakdown['indicator_score']}/3")
        print(f" Context Score: {breakdown['context_score']}/2")
        print(f" Risk Score: {breakdown['risk_score']}/2")
        print(f" {'─' * 30}")
        print(f" Total Score: {breakdown['confidence_score']}/10")
        print()

        # Entry/Exit levels
        print("Entry/Exit Levels:")
        print(f" Entry Price: ₹{breakdown['entry_price']:.2f}")
        print(f" Stop Loss: ₹{breakdown['stop_loss']:.2f}")
        print(f" Target Price: ₹{breakdown['target_price']:.2f}")
        print(f" RRR: {breakdown['rrr']:.2f}:1 {'✅' if breakdown['rrr'] >= 1.5 else '❌'}")
        print()

        # Confirmation indicators
        print("Indicator Confirmation:")
        print(f" Supporting: {', '.join(breakdown['supporting_indicators']) if breakdown['supporting_indicators'] else 'None'}")
        print(f" Opposing: {', '.join(breakdown['opposing_indicators']) if breakdown['opposing_indicators'] else 'None'}")
        print()

        # Historical context
        print("Historical Context:")
        print(f" Pattern Win Rate: {breakdown['historical_win_rate']*100:.0f}%")
        print(f" Market Regime: {breakdown['market_regime']}")
        print(f" RSI: {breakdown['rsi_value']:.1f}")
        print(f" MACD Signal: {breakdown['macd_signal']}")
        print()


class PerformanceTracker:
    """
    Track and report on signal performance
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize tracker

        Args:
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.signals: List[SignalRecord] = []
        self.daily_metrics: List[PerformanceMetrics] = []

    def record_signal(self, signal: SignalRecord) -> None:
        """
        Record a new signal

        Args:
            signal: SignalRecord to add
        """
        self.signals.append(signal)
        self.logger.debug(f"Signal recorded: {signal.symbol} {signal.direction} ({signal.pattern})")

    def close_signal(
        self,
        symbol: str,
        close_price: float,
        close_timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Close an open signal position

        Args:
            symbol: Stock symbol
            close_price: Price at which position closed
            close_timestamp: When position closed

        Returns:
            True if signal was closed
        """
        if close_timestamp is None:
            close_timestamp = datetime.now()

        # Find matching open signal
        open_signals = [s for s in self.signals if s.symbol == symbol and s.status == "OPEN"]
        if not open_signals:
            self.logger.warning(f"No open signal found for {symbol}")
            return False

        signal = open_signals[0]  # Get first open signal

        # Calculate P&L
        if signal.direction == "BUY":
            pnl_pct = ((close_price - signal.entry_price) / signal.entry_price) * 100
        else:
            pnl_pct = ((signal.entry_price - close_price) / signal.entry_price) * 100

        # Determine status
        if pnl_pct > 0:
            status = "CLOSED_WIN"
        elif pnl_pct < 0:
            status = "CLOSED_LOSS"
        else:
            status = "CLOSED_BREAK_EVEN"

        # Update signal
        signal.close_price = close_price
        signal.close_timestamp = close_timestamp
        signal.pnl_pct = pnl_pct
        signal.status = status
        self.logger.info(f"Signal closed: {symbol} {status} (P&L: {pnl_pct:+.2f}%)")

        return True

    def get_today_statistics(self) -> PerformanceMetrics:
        """
        Calculate today's performance statistics

        Returns:
            PerformanceMetrics with today's stats
        """
        today = datetime.now().date()
        today_signals = [s for s in self.signals if s.timestamp.date() == today]

        metrics = PerformanceMetrics()
        metrics.signals_generated = len(today_signals)
        metrics.signals_sent = len([s for s in today_signals if s.tier in ['PREMIUM', 'HIGH', 'MEDIUM']])
        metrics.signals_open = len([s for s in today_signals if s.status == "OPEN"])
        metrics.signals_closed = len([s for s in today_signals if s.status.startswith("CLOSED")])

        # Count by tier
        for signal in today_signals:
            metrics.signals_by_tier[signal.tier] = metrics.signals_by_tier.get(signal.tier, 0) + 1

        # Calculate performance on closed signals
        closed_signals = [s for s in today_signals if s.status.startswith("CLOSED")]
        if closed_signals:
            wins = len([s for s in closed_signals if s.status == "CLOSED_WIN"])
            losses = len([s for s in closed_signals if s.status == "CLOSED_LOSS"])
            breakeven = len([s for s in closed_signals if s.status == "CLOSED_BREAK_EVEN"])

            metrics.closed_wins = wins
            metrics.closed_losses = losses
            metrics.closed_breakeven = breakeven

            metrics.win_rate = wins / len(closed_signals) if closed_signals else 0
            metrics.avg_rrr_promised = sum(s.rrr for s in closed_signals) / len(closed_signals) if closed_signals else 0

            # Calculate actual RRR achieved
            total_profit = sum(s.pnl_pct for s in closed_signals if s.status == "CLOSED_WIN")
            total_loss = abs(sum(s.pnl_pct for s in closed_signals if s.status == "CLOSED_LOSS"))

            metrics.avg_rrr_achieved = total_profit / total_loss if total_loss > 0 else 0
            metrics.profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
            metrics.total_pnl_pct = sum(s.pnl_pct for s in closed_signals)

        return metrics

    def get_signal_history(self, n_days: int = 7) -> List[SignalRecord]:
        """
        Get signal history for past N days

        Args:
            n_days: Number of days to retrieve

        Returns:
            List of SignalRecord objects
        """
        cutoff_date = datetime.now().date() - timedelta(days=n_days)
        return [s for s in self.signals if s.timestamp.date() >= cutoff_date]

    def export_signals_json(self, filepath: str = "signals_history.json") -> None:
        """
        Export signal history to JSON

        Args:
            filepath: Path to export file
        """
        signals_data = []
        for signal in self.signals:
            signals_data.append({
                'timestamp': signal.timestamp.isoformat(),
                'symbol': signal.symbol,
                'direction': signal.direction,
                'pattern': signal.pattern,
                'tier': signal.tier,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'target_price': signal.target_price,
                'rrr': signal.rrr,
                'win_rate': signal.win_rate,
                'status': signal.status,
                'close_price': signal.close_price,
                'close_timestamp': signal.close_timestamp.isoformat() if signal.close_timestamp else None,
                'pnl_pct': signal.pnl_pct
            })
        with open(filepath, 'w') as f:
            json.dump(signals_data, f, indent=2)
        self.logger.info(f"✓ Exported {len(signals_data)} signals to {filepath}")


class DashboardInterface:
    """
    Interactive dashboard interface

    Manages user interaction and display updates
    """

    def __init__(
        self,
        config: BotConfiguration,
        analyzer: MarketAnalyzer,
        validator: SignalValidator,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize dashboard interface

        Args:
            config: BotConfiguration instance
            analyzer: MarketAnalyzer instance
            validator: SignalValidator instance
            logger: Optional logger
        """
        self.config = config
        self.dashboard = MonitoringDashboard(config, analyzer, validator, logger)
        self.adhoc_validator = AdhocSignalValidator(config, analyzer, validator, logger)
        self.tracker = PerformanceTracker(logger)
        self.logger = logger or logging.getLogger(__name__)

    def run_interactive_mode(self) -> None:
        """Run interactive dashboard mode"""
        self.logger.info("Starting interactive dashboard mode")

        while True:
            try:
                # Get user command
                command = input("\nCommand [d]ashboard [v]alidate [h]istory [s]tats [q]uit: ").strip().lower()

                if command == 'q' or command == 'quit':
                    self.logger.info("Exiting dashboard")
                    break
                elif command == 'd' or command == 'dashboard':
                    # Would normally fetch live data here
                    # For now, just show sample
                    sample_data = [{
                        'valid': True,
                        'symbol': 'INFY',
                        'price': 1650.50,
                        'patterns': [],
                        'market_regime': MarketRegime.UPTREND,
                        'indicators': None
                    }]
                    self.dashboard.display_dashboard(sample_data)

                elif command == 'v' or command == 'validate':
                    self._handle_manual_validation()

                elif command == 'h' or command == 'history':
                    self._display_signal_history()

                elif command == 's' or command == 'stats':
                    metrics = self.tracker.get_today_statistics()
                    self._display_metrics(metrics)

                else:
                    print("Invalid command. Try again.")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                self.logger.error(f"Error in interactive mode: {str(e)}")

    def _handle_manual_validation(self) -> None:
        """Handle manual signal validation"""
        print("\n📊 MANUAL SIGNAL VALIDATION")
        print("─" * 50)

        symbol = input("Enter stock symbol (e.g., INFY): ").strip().upper()
        direction = input("Signal direction (BUY/SELL): ").strip().upper()
        pattern = input("Pattern name: ").strip()

        # NOTE: Real implementation would fetch live data from Upstox API or another source
        print(f"\nValidating {symbol} {direction} signal ({pattern})...")
        print("Note: Real implementation would fetch live data from Upstox API")

        # Sample validation result (replace with real validation call)
        sample_breakdown = {
            'valid': True,
            'validation_passed': True,
            'symbol': symbol,
            'direction': direction,
            'pattern': pattern,
            'signal_tier': 'HIGH',
            'confidence_score': 7,
            'pattern_score': 2,
            'indicator_score': 3,
            'context_score': 1,
            'risk_score': 1,
            'patterns_detected': [pattern],
            'supporting_indicators': ['RSI', 'MACD'],
            'opposing_indicators': [],
            'entry_price': 1650.50,
            'stop_loss': 1640.00,
            'target_price': 1680.00,
            'rrr': 2.0,
            'historical_win_rate': 0.60,
            'market_regime': 'UPTREND',
            'rsi_value': 28.5,
            'macd_signal': 'BULLISH'
        }

        self.adhoc_validator.display_validation_result(sample_breakdown)

    def _display_signal_history(self) -> None:
        """Display signal history"""
        print("\n📋 SIGNAL HISTORY (Last 7 Days)")
        print("─" * 70)
        signals = self.tracker.get_signal_history(n_days=7)
        if not signals:
            print("No signals in history")
            return

        for signal in signals[-10:]:  # Show last 10
            emoji = "🟢" if signal.direction == "BUY" else "🔴"
            status = "✅" if signal.status == "CLOSED_WIN" else "❌" if signal.status == "CLOSED_LOSS" else "⏳"
            time_str = signal.timestamp.strftime("%m-%d %H:%M")
            pnl_str = f"{signal.pnl_pct:+.2f}%" if signal.pnl_pct is not None else "Open"
            print(
                f"{emoji} {signal.symbol:8} {signal.direction:4} {signal.pattern:20} "
                f"{signal.tier:7} {pnl_str:>8} {status} {time_str}"
            )

    def _display_metrics(self, metrics: PerformanceMetrics) -> None:
        """Display performance metrics"""
        print("\n📊 TODAY'S PERFORMANCE METRICS")
        print("─" * 70)
        print(
            f"Generated: {metrics.signals_generated:3} | "
            f"Sent: {metrics.signals_sent:3} | "
            f"Open: {metrics.signals_open:3} | "
            f"Closed: {metrics.signals_closed:3}"
        )
        print(
            f"Win Rate: {metrics.win_rate*100:5.1f}% | "
            f"Wins: {metrics.closed_wins:2} | "
            f"Losses: {metrics.closed_losses:2} | "
            f"P&L: {metrics.total_pnl_pct:+6.2f}%"
        )
        print(
            f"Profit Factor: {metrics.profit_factor:.2f}x | "
            f"Avg RRR: {metrics.avg_rrr_achieved:.2f}:1"
        )


if __name__ == "__main__":
    # Test the dashboard
    from config import get_config

    config = get_config()
    analyzer = MarketAnalyzer(config)
    validator = SignalValidator(config)

    interface = DashboardInterface(config, analyzer, validator)
    print("✓ Dashboard initialized successfully")
    print("Ready to run: interface.run_interactive_mode()")
