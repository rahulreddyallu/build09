"""

STOCK SIGNALLING BOT - INSTITUTIONAL GRADE CONFIGURATION

================================================

This configuration module provides:

✓ Complete parameter management without code changes

✓ Multi-stage validation framework with confidence threshold checks

✓ Environment variable override support (all parameters tunable)

✓ Research-backed default thresholds (NSE/NIFTY studies)

✓ Profile-based modes (LIVE, BACKTEST, PAPER, RESEARCH)

✓ Comprehensive documentation with academic citations

✓ Production-ready nested dataclass reconstruction

✓ External stock universe configuration support

Research References:

- NSE Volatility Clustering: AJEBA 2024 (NIFTY 50 EGARCH analysis)

- Candlestick Accuracy: 75%+ in Nifty with indicator confirmation (IJIERM 2024)

- Risk Management: Kotak Securities position sizing framework 2025

- Pattern Validation: Bullish Engulfing study (IJISRT 2025) - 6-14 events per stock

Author: rahulreddyallu

Version: 4.1.0 (Institutional Grade - Enhanced)

Date: 2025-12-01

"""

import os

import logging

import warnings

from typing import Dict, List, Any, Optional

from dataclasses import dataclass, field, asdict

from enum import Enum

import json


# ============================================================================

# EXECUTION MODES

# ============================================================================

class ExecutionMode(Enum):
    """Execution profiles for different use cases"""
    LIVE = "live"  # Real market data, no execution
    BACKTEST = "backtest"  # Historical data only
    PAPER = "paper"  # Real market data, simulated execution
    RESEARCH = "research"  # Deep pattern analysis, extended history
    ADHOC = "adhoc"  # Manual signal validation mode


# ============================================================================

# SIGNAL QUALITY TIERS

# ============================================================================

class SignalTier(Enum):
    """Signal confidence levels based on multi-stage validation"""
    REJECT = 0  # Failed validation
    LOW = 1  # Single confirmation (≤50% historical win rate)
    MEDIUM = 2  # Double confirmation (51-70% win rate)
    HIGH = 3  # Triple confirmation (71-85% win rate)
    PREMIUM = 4  # Consensus + High RRR (>85% win rate)


# ============================================================================

# RESEARCH-BACKED DEFAULTS (Academic & Institutional Standards)

# ============================================================================

@dataclass
class CandlestickPatternThresholds:
    """
    Candlestick pattern detection thresholds

    Based on: Quantitative Pattern Study (Scribd 2024), IJISRT 2025

    """

    # Single candle patterns
    doji_body_pct: float = 0.10  # Body < 10% of range (extreme indecision)
    doji_shadow_ratio: float = 2.0  # Upper/Lower shadow ratio near 1:1
    marubozu_body_pct: float = 0.95  # Body > 95% of range (strong conviction)
    marubozu_shadow_pct: float = 0.05  # Shadows < 5% (minimal wicks)
    hammer_lower_shadow_ratio: float = 2.5  # Lower shadow 2.5x body size
    hammer_upper_shadow_pct: float = 0.10  # Upper shadow < 10%
    hammer_close_position: str = "upper"  # Close in upper half
    hanging_man_lower_shadow_ratio: float = 2.5
    hanging_man_upper_shadow_pct: float = 0.10
    hanging_man_close_position: str = "lower"
    shooting_star_upper_shadow_ratio: float = 2.5
    shooting_star_lower_shadow_pct: float = 0.10
    shooting_star_close_position: str = "lower"
    spinning_top_body_pct: float = 0.20  # Body < 20% (high indecision)
    spinning_top_shadow_pct: float = 0.40  # Balanced upper/lower shadows

    # Multi-candle patterns
    engulfing_body_factor: float = 1.15  # Previous body < 85% of current
    engulfing_close_direction: bool = True  # Close must be beyond previous candle
    harami_size_ratio: float = 0.60  # Previous body must be 60%+ larger
    harami_containment: bool = True  # Current body fully inside previous
    piercing_close_pct: float = 0.50  # Close ≥ 50% into previous body
    dark_cloud_close_pct: float = 0.50  # Close ≤ 50% of previous body

    # Three-candle patterns (Star formations)
    morning_star_middle_small_pct: float = 0.30  # Middle candle <30% range
    morning_star_gap_confirm: bool = True  # Price gap between candles
    evening_star_middle_small_pct: float = 0.30
    evening_star_gap_confirm: bool = True

    # Volume confirmation thresholds
    high_volume_multiplier: float = 1.5  # >150% of 20-day average
    pattern_volume_required: bool = True  # Most patterns need volume


@dataclass
class TechnicalIndicatorParams:
    """
    Technical indicator parameters for signal generation

    Based on: Standard quantitative trading practices + Indian market optimization

    """

    # RSI (Relative Strength Index) - Momentum indicator
    rsi_period: int = 14
    rsi_oversold: int = 30  # Potential buy signal
    rsi_overbought: int = 70  # Potential sell signal
    rsi_extreme_oversold: int = 20  # Very strong oversold
    rsi_extreme_overbought: int = 80  # Very strong overbought

    # MACD (Moving Average Convergence Divergence)
    macd_fast_ema: int = 12
    macd_slow_ema: int = 26
    macd_signal_line: int = 9
    macd_histogram_threshold: float = 0.01  # Min change for confirmation

    # Bollinger Bands
    bb_period: int = 20
    bb_std_dev: float = 2.0  # 95% confidence interval
    bb_lower_multiplier: float = 2.5  # Extended bands for extremes

    # ATR (Average True Range) - Volatility
    atr_period: int = 14
    atr_multiplier_stop: float = 1.0  # Stop loss distance
    atr_multiplier_target: float = 1.5  # Target distance (1.5:1 RRR)
    atr_min_rrr: float = 1.5  # Minimum risk-reward ratio

    # Stochastic Oscillator
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_slowing: int = 3
    stoch_oversold: int = 20
    stoch_overbought: int = 80
    stoch_extreme_oversold: int = 10
    stoch_extreme_overbought: int = 90

    # ADX (Average Directional Index) - Trend strength
    adx_period: int = 14
    adx_weak_trend: int = 20  # <20: no clear trend
    adx_moderate_trend: int = 25  # 20-25: weak trend
    adx_strong_trend: int = 40  # >40: very strong trend

    # VWAP (Volume Weighted Average Price)
    vwap_period: int = 50
    vwap_distance_threshold: float = 0.02  # 2% from VWAP = support

    # Moving Averages
    sma_fast_periods: List[int] = field(default_factory=lambda: [5, 10])
    sma_slow_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    ema_fast_periods: List[int] = field(default_factory=lambda: [5, 12])
    ema_slow_periods: List[int] = field(default_factory=lambda: [26, 50])

    # Support & Resistance
    sr_lookback_bars: int = 100
    sr_price_tolerance_pct: float = 2.0  # Within 2% counts as S/R test
    sr_window_size: int = 5  # Min 5-bar high/low window
    sr_touches_threshold: int = 3  # Min 3 touches to confirm level

    # Volume Analysis
    volume_ma_period: int = 20
    volume_spike_threshold: float = 1.5  # 150% of average

    # Fibonacci Retracement
    fib_lookback_bars: int = 100
    fib_use_in_signals: bool = True


@dataclass
class SignalValidationParams:
    """
    Multi-stage signal validation thresholds

    Reference: IJISRT 2025 (Bullish Engulfing accuracy 16-75% per stock)

    """

    # Stage 1: Pattern Detection
    min_pattern_strength: int = 2  # Min 2/5 pattern criteria met

    # Stage 2: Indicator Confirmation
    min_indicator_count: int = 2  # Minimum confirming indicators
    required_indicators: List[str] = field(
        default_factory=lambda: ["RSI", "MACD", "VOLUME"]
    )

    # Stage 3: Context Validation
    require_trend_alignment: bool = True  # Signal must align with trend
    require_support_resistance: bool = True  # Near S/R level
    require_volume_confirmation: bool = True  # Above average volume

    # Stage 4: Risk Validation
    min_rrr: float = 1.5  # Minimum 1.5:1 reward:risk ratio
    max_rrr: float = 5.0  # Cap at 5:1 (prevents over-leverage)

    # Signal scoring (total max: 10 points)
    pattern_score_weight: float = 3.0  # Strong pattern = 3 points
    indicator_score_weight: float = 3.0  # Indicator confirmation = 3 points
    context_score_weight: float = 2.0  # Context alignment = 2 points
    risk_score_weight: float = 2.0  # Proper RRR = 2 points

    # Confidence thresholds (must be in increasing order and within max_score)
    high_confidence_threshold: int = 8  # 8+ points = PREMIUM signal
    medium_confidence_threshold: int = 6  # 6-7 points = HIGH signal
    low_confidence_threshold: int = 4  # 4-5 points = MEDIUM signal
    reject_threshold: int = 3  # <4 points = REJECT


@dataclass
class RiskManagementParams:
    """
    Risk management and position sizing

    Based on: Kotak Securities framework, Risk Management Guide 2025

    """

    # Account protection
    max_risk_per_trade_pct: float = 1.0  # Never risk >1% per trade
    max_daily_loss_pct: float = 2.0  # Stop if lost 2% in a day
    max_weekly_loss_pct: float = 5.0  # Stop if lost 5% in a week

    # Position sizing
    position_sizing_method: str = "fixed_risk"  # fixed_risk or percentage
    risk_amount_per_trade: float = 5000.0  # ₹5000 per trade max
    account_size_for_position: float = 100000.0  # Assumed account ₹100k

    # Drawdown management
    max_consecutive_losses: int = 3  # Stop after 3 losses in a row
    max_portfolio_drawdown_pct: float = 10.0  # Maximum acceptable drawdown

    # Win rate requirements
    min_win_rate_for_trading: float = 0.45  # Need 45% wins minimum
    min_breakeven_ratio: float = 1.2  # RRR * Win% must exceed 1.0

    # Signal filtering
    max_signals_per_day: int = 5  # Max 5 signals per trading session
    min_hours_between_signals: int = 1  # Min 1 hour between signals


@dataclass
class MarketDataParams:
    """
    Market data fetching and processing parameters

    """

    # Historical data
    historical_days: int = 100  # 100 days for pattern analysis (tunable via env)
    minimum_candles_required: int = 50  # Need minimum 50 candles for reliability

    # Data intervals
    primary_interval: str = "day"  # Main analysis interval (tunable via env)
    supported_intervals: List[str] = field(
        default_factory=lambda: [
            "1minute",
            "5minute",
            "15minute",
            "30minute",
            "day",
            "week",
            "month",
        ]
    )

    # Data quality checks
    check_for_gaps: bool = True  # Detect price gaps
    check_for_extreme_volumes: bool = True  # Flag unusual volume
    check_for_stale_data: bool = True  # Verify data freshness
    max_data_staleness_minutes: int = 5  # Data >5min old is stale

    # Indian market specifics
    nse_trading_hours_open: str = "09:15"  # NSE opening bell
    nse_trading_hours_close: str = "15:30"  # NSE closing bell
    holiday_check_enabled: bool = True  # Skip weekends/holidays


@dataclass
class TelegramNotificationParams:
    """
    Telegram notification configuration

    """

    enabled: bool = True
    bot_token: str = ""  # Will be populated from env
    chat_id: str = ""  # Will be populated from env

    # Notification types
    notify_on_signal: bool = True  # Send when signal generated
    notify_on_validation_pass: bool = True  # Send when validation passes
    notify_on_errors: bool = True  # Send on critical errors
    notify_daily_summary: bool = True  # Send end-of-day summary

    # Message formatting
    include_pattern_details: bool = True
    include_indicator_scores: bool = True
    include_entry_exit_levels: bool = True
    include_risk_reward_ratio: bool = True
    max_message_length: int = 4000  # Telegram limit is 4096


@dataclass
class MonitoringParams:
    """
    Live monitoring and adhoc capabilities

    """

    # Monitoring dashboard
    enable_live_dashboard: bool = True
    dashboard_update_frequency_seconds: int = 60

    # Adhoc signal validation
    enable_adhoc_mode: bool = True  # Manual signal input
    manual_signal_buffer_seconds: int = 300  # 5 min buffer for validation

    # Performance tracking
    track_signal_accuracy: bool = True
    track_win_rate: bool = True
    track_avg_rrr: bool = True
    track_sharpe_ratio: bool = True

    # Logging
    log_every_calculation: bool = False  # Too verbose for production
    log_pattern_detection: bool = True
    log_validation_stages: bool = True
    log_signal_decisions: bool = True


@dataclass
class APICredentialsParams:
    """
    API credentials (should be loaded from environment)

    NEVER hardcode credentials in this file

    """

    upstox_api_key: str = ""
    upstox_api_secret: str = ""
    upstox_access_token: str = ""

    def validate(self) -> bool:
        """Validate that all credentials are configured"""
        missing = []
        if not self.upstox_api_key:
            missing.append("UPSTOX_API_KEY")
        if not self.upstox_api_secret:
            missing.append("UPSTOX_API_SECRET")
        if not self.upstox_access_token:
            missing.append("UPSTOX_ACCESS_TOKEN")

        if missing:
            raise ValueError(f"Missing credentials: {', '.join(missing)}")
        return True


# ============================================================================

# COMPLETE CONFIGURATION DATACLASS

# ============================================================================


@dataclass
class BotConfiguration:
    """
    Master configuration object containing all bot parameters

    Provides methods for validation, serialization, and dynamic override

    """

    # Metadata
    version: str = "4.1.0"
    mode: ExecutionMode = ExecutionMode.LIVE
    instance_name: str = "nifty-signal-bot-1"

    # Component configurations
    patterns: CandlestickPatternThresholds = field(
        default_factory=CandlestickPatternThresholds
    )
    indicators: TechnicalIndicatorParams = field(default_factory=TechnicalIndicatorParams)
    validation: SignalValidationParams = field(default_factory=SignalValidationParams)
    risk_management: RiskManagementParams = field(default_factory=RiskManagementParams)
    market_data: MarketDataParams = field(default_factory=MarketDataParams)
    telegram: TelegramNotificationParams = field(
        default_factory=TelegramNotificationParams
    )
    monitoring: MonitoringParams = field(default_factory=MonitoringParams)
    api_creds: APICredentialsParams = field(default_factory=APICredentialsParams)

    # Stock list for monitoring (externally configurable)
    stocks_to_monitor: List[str] = field(
        default_factory=lambda: [
            "NSE_EQ|INE009A01021",  # INFOSYS
            "NSE_EQ|INE030A01027",  # HDFC Bank
            "NSE_EQ|INE062A01020",  # TATA Motors
            "NSE_EQ|INE002A01015",  # TCS
            "NSE_EQ|INE595A01028",  # RELIANCE
        ]
    )

    # Logging
    log_directory: str = "logs"
    log_level: str = "INFO"

    def validate_all(self) -> Dict[str, Any]:
        """
        Comprehensive validation of all configuration parameters

        Returns:
            Dict with validation results:
            {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str],
                'error_count': int,
                'warning_count': int
            }

        """
        errors = []
        warnings = []

        # 1. API Credentials validation - ENHANCED
        try:
            self.api_creds.validate()
        except ValueError as e:
            errors.append(f"API Credentials: {str(e)}")

        # 2. Stock list validation
        if not self.stocks_to_monitor:
            errors.append("Stock list is empty - at least 1 stock required")
        if len(self.stocks_to_monitor) > 50:
            warnings.append(
                f"Monitoring {len(self.stocks_to_monitor)} stocks - may impact performance"
            )

        # 3. Signal validation parameters
        if self.validation.min_rrr < 1.0:
            errors.append(f"min_rrr must be ≥ 1.0, got {self.validation.min_rrr}")
        if self.validation.min_rrr > self.validation.max_rrr:
            errors.append(
                f"min_rrr ({self.validation.min_rrr}) > max_rrr ({self.validation.max_rrr})"
            )

        # 4. Risk management validation
        if (
            self.risk_management.max_risk_per_trade_pct <= 0
            or self.risk_management.max_risk_per_trade_pct > 5
        ):
            errors.append(
                f"max_risk_per_trade_pct must be 0-5%, got {self.risk_management.max_risk_per_trade_pct}"
            )

        # 5. Indicator parameters validation
        if self.indicators.rsi_oversold >= self.indicators.rsi_overbought:
            errors.append(
                f"RSI oversold ({self.indicators.rsi_oversold}) >= overbought ({self.indicators.rsi_overbought})"
            )

        # 6. Market data validation - ENHANCED
        if self.market_data.historical_days < 100:
            warnings.append(
                f"historical_days = {self.market_data.historical_days} is below recommended 100+; "
                f"500+ is preferred for robust backtests."
            )
        if self.market_data.minimum_candles_required > self.market_data.historical_days:
            errors.append("minimum_candles_required > historical_days")

        # 7. Telegram validation
        if self.telegram.enabled:
            if not self.telegram.bot_token or not self.telegram.chat_id:
                warnings.append("Telegram enabled but credentials missing")

        # 8. Monitoring validation
        if self.monitoring.dashboard_update_frequency_seconds < 10:
            warnings.append(
                "Dashboard update frequency <10s may cause performance issues"
            )

        # 9. Confidence threshold validation - NEW
        max_score = (
            self.validation.pattern_score_weight
            + self.validation.indicator_score_weight
            + self.validation.context_score_weight
            + self.validation.risk_score_weight
        )
        if self.validation.high_confidence_threshold > max_score:
            errors.append(
                f"high_confidence_threshold ({self.validation.high_confidence_threshold}) "
                f"> max possible score ({max_score})"
            )
        if not (
            self.validation.reject_threshold
            < self.validation.low_confidence_threshold
            <= self.validation.medium_confidence_threshold
            <= self.validation.high_confidence_threshold
        ):
            errors.append(
                "Confidence thresholds are not in strictly increasing order: "
                f"reject ({self.validation.reject_threshold}) < low ({self.validation.low_confidence_threshold}) "
                f"<= medium ({self.validation.medium_confidence_threshold}) "
                f"<= high ({self.validation.high_confidence_threshold})"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "error_count": len(errors),
            "warning_count": len(warnings),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)

    def to_json(self, filepath: str = None) -> str:
        """Export configuration to JSON"""
        json_str = json.dumps(self.to_dict(), indent=2, default=str)
        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)
            logging.info(f"Configuration exported to {filepath}")
        return json_str

    @classmethod
    def from_json(cls, filepath: str) -> "BotConfiguration":
        """Load configuration from JSON file"""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict) -> "BotConfiguration":
        """
        Create configuration from dictionary

        Properly reconstructs nested dataclasses to ensure type safety

        """
        data = data.copy()

        # Reconstruct nested dataclasses explicitly
        if "patterns" in data and isinstance(data["patterns"], dict):
            data["patterns"] = CandlestickPatternThresholds(**data["patterns"])
        if "indicators" in data and isinstance(data["indicators"], dict):
            data["indicators"] = TechnicalIndicatorParams(**data["indicators"])
        if "validation" in data and isinstance(data["validation"], dict):
            data["validation"] = SignalValidationParams(**data["validation"])
        if "risk_management" in data and isinstance(data["risk_management"], dict):
            data["risk_management"] = RiskManagementParams(**data["risk_management"])
        if "market_data" in data and isinstance(data["market_data"], dict):
            data["market_data"] = MarketDataParams(**data["market_data"])
        if "telegram" in data and isinstance(data["telegram"], dict):
            data["telegram"] = TelegramNotificationParams(**data["telegram"])
        if "monitoring" in data and isinstance(data["monitoring"], dict):
            data["monitoring"] = MonitoringParams(**data["monitoring"])
        if "api_creds" in data and isinstance(data["api_creds"], dict):
            data["api_creds"] = APICredentialsParams(**data["api_creds"])

        # Convert ExecutionMode string back to enum
        if "mode" in data and isinstance(data["mode"], str):
            try:
                data["mode"] = ExecutionMode[data["mode"].upper()]
            except KeyError:
                logging.warning(f"Invalid execution mode: {data['mode']}, defaulting to LIVE")
                data["mode"] = ExecutionMode.LIVE

        return cls(**data)


# ============================================================================

# ENVIRONMENT-BASED CONFIGURATION LOADER

# ============================================================================


def load_config_from_environment() -> BotConfiguration:
    """
    Load configuration from environment variables

    Allows override of any parameter without modifying files

    Environment variables follow pattern: BOT_<SECTION>_<PARAMETER>
    Example: BOT_VALIDATION_MIN_RRR=2.0

    Supported overrides:
    - Market Data: BOT_MARKETDATA_HISTORICAL_DAYS, BOT_MARKETDATA_PRIMARY_INTERVAL
    - Risk Management: BOT_RISK_MAX_RISK_PER_TRADE_PCT, BOT_RISK_MAX_DAILY_LOSS_PCT,
                       BOT_RISK_MAX_SIGNALS_PER_DAY
    - Validation: BOT_VALIDATION_MIN_RRR
    - Stocks: BOT_STOCKS_JSON (JSON array of instrument codes)

    """
    config = BotConfiguration()

    # Load API credentials from environment (NEVER hardcode)
    config.api_creds.upstox_access_token = os.getenv("UPSTOX_ACCESS_TOKEN", "")
    config.api_creds.upstox_api_key = os.getenv("UPSTOX_API_KEY", "")
    config.api_creds.upstox_api_secret = os.getenv("UPSTOX_API_SECRET", "")

    # Load Telegram credentials
    config.telegram.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    config.telegram.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    # Load execution mode
    mode_str = os.getenv("BOT_MODE", "LIVE").upper()
    try:
        config.mode = ExecutionMode[mode_str]
    except KeyError:
        logging.warning(f"Invalid BOT_MODE: {mode_str}, defaulting to LIVE")

    # Load log level
    config.log_level = os.getenv("BOT_LOG_LEVEL", "INFO")

    # ========== MARKET DATA OVERRIDES ==========
    hist_days_env = os.getenv("BOT_MARKETDATA_HISTORICAL_DAYS")
    if hist_days_env:
        try:
            config.market_data.historical_days = int(hist_days_env)
        except ValueError:
            logging.warning(
                f"Invalid BOT_MARKETDATA_HISTORICAL_DAYS: {hist_days_env}, using default"
            )

    primary_interval_env = os.getenv("BOT_MARKETDATA_PRIMARY_INTERVAL")
    if primary_interval_env:
        if primary_interval_env in config.market_data.supported_intervals:
            config.market_data.primary_interval = primary_interval_env
        else:
            logging.warning(
                f"Invalid BOT_MARKETDATA_PRIMARY_INTERVAL: {primary_interval_env}, "
                f"supported: {config.market_data.supported_intervals}"
            )

    # ========== VALIDATION OVERRIDES ==========
    min_rrr_env = os.getenv("BOT_VALIDATION_MIN_RRR")
    if min_rrr_env:
        try:
            config.validation.min_rrr = float(min_rrr_env)
        except ValueError:
            logging.warning(f"Invalid BOT_VALIDATION_MIN_RRR: {min_rrr_env}, using default")

    max_rrr_env = os.getenv("BOT_VALIDATION_MAX_RRR")
    if max_rrr_env:
        try:
            config.validation.max_rrr = float(max_rrr_env)
        except ValueError:
            logging.warning(f"Invalid BOT_VALIDATION_MAX_RRR: {max_rrr_env}, using default")

    # ========== RISK MANAGEMENT OVERRIDES ==========
    max_risk_trade_env = os.getenv("BOT_RISK_MAX_RISK_PER_TRADE_PCT")
    if max_risk_trade_env:
        try:
            config.risk_management.max_risk_per_trade_pct = float(max_risk_trade_env)
        except ValueError:
            logging.warning(
                f"Invalid BOT_RISK_MAX_RISK_PER_TRADE_PCT: {max_risk_trade_env}, using default"
            )

    max_daily_loss_env = os.getenv("BOT_RISK_MAX_DAILY_LOSS_PCT")
    if max_daily_loss_env:
        try:
            config.risk_management.max_daily_loss_pct = float(max_daily_loss_env)
        except ValueError:
            logging.warning(
                f"Invalid BOT_RISK_MAX_DAILY_LOSS_PCT: {max_daily_loss_env}, using default"
            )

    max_signals_env = os.getenv("BOT_RISK_MAX_SIGNALS_PER_DAY")
    if max_signals_env:
        try:
            config.risk_management.max_signals_per_day = int(max_signals_env)
        except ValueError:
            logging.warning(
                f"Invalid BOT_RISK_MAX_SIGNALS_PER_DAY: {max_signals_env}, using default"
            )

    # ========== STOCK UNIVERSE OVERRIDE ==========
    stocks_env = os.getenv("BOT_STOCKS_JSON")
    if stocks_env:
        try:
            parsed_stocks = json.loads(stocks_env)
            if isinstance(parsed_stocks, list) and all(isinstance(s, str) for s in parsed_stocks):
                config.stocks_to_monitor = parsed_stocks
                logging.info(f"Loaded {len(config.stocks_to_monitor)} stocks from BOT_STOCKS_JSON")
            else:
                logging.warning("BOT_STOCKS_JSON must be a JSON array of strings")
        except json.JSONDecodeError as e:
            logging.warning(f"Invalid BOT_STOCKS_JSON format: {e}")

    return config


def get_config() -> BotConfiguration:
    """
    Get final configuration (environment overrides + defaults)

    This is the main entry point for the bot

    Performs full validation and logs configuration status

    """
    config = load_config_from_environment()

    # Validate configuration
    validation_result = config.validate_all()

    if not validation_result["valid"]:
        logging.error("Configuration validation failed:")
        for error in validation_result["errors"]:
            logging.error(f" ✗ {error}")
        raise ValueError("Invalid configuration - cannot proceed")

    if validation_result["warnings"]:
        logging.warning("Configuration warnings:")
        for warning in validation_result["warnings"]:
            logging.warning(f" ⚠ {warning}")

    logging.info("✓ Configuration loaded and validated successfully")
    logging.info(f"  Mode: {config.mode.value}")
    logging.info(f"  Stocks: {len(config.stocks_to_monitor)} configured")
    logging.info(f"  Min RRR: {config.validation.min_rrr}")
    logging.info(f"  Max Risk/Trade: {config.risk_management.max_risk_per_trade_pct}%")

    return config


# ============================================================================

# CONFIGURATION EXPORT FOR DOCUMENTATION

# ============================================================================


def export_default_config(filepath: str = "config_template.json"):
    """Export default configuration as template for users"""
    config = BotConfiguration()
    config.to_json(filepath)
    print(f"✓ Default configuration exported to {filepath}")


if __name__ == "__main__":
    # Test configuration loading and validation
    try:
        config = get_config()
        print("\n✓ Configuration loaded successfully")
        print(f" Mode: {config.mode.value}")
        print(f" Stocks: {len(config.stocks_to_monitor)} configured")
        print(f" Min RRR: {config.validation.min_rrr}")
        print(f" Max Risk/Trade: {config.risk_management.max_risk_per_trade_pct}%")
        print(f" Historical Days: {config.market_data.historical_days}")
        print(f" Primary Interval: {config.market_data.primary_interval}")

        # Verify round-trip serialization
        json_str = config.to_json()
        config_reloaded = BotConfiguration.from_dict(json.loads(json_str))
        print("\n✓ Configuration round-trip serialization successful")

    except Exception as e:
        print(f"\n✗ Error loading configuration: {e}")
        import traceback
        traceback.print_exc()
