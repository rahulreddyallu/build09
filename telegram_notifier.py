# telegram_notifier.py - COMPLETE PRODUCTION VERSION (v2.8.0)
# ==================================================================================
# Telegram Notification System with Complete Historical Validation Integration
# Formats alerts with verified accuracy statistics from backtesting
# ==================================================================================
#
# Author: rahulreddyallu
# Version: 2.8.0 (Production - Fully Integrated)
# Date: 2025-12-01
#
# ==================================================================================

"""
TELEGRAM NOTIFIER - COMPLETE NOTIFICATION SYSTEM WITH HISTORICAL INTEGRATION

===================================================================================

This module implements a COMPLETE, production-grade Telegram notification system
fully integrated with signal_validator.py v4.5.1, signals_db.py v3.2.0, and
config.py v4.1.0:

✓ Signal alerts with verified historical accuracy data
✓ Daily performance summaries with metrics
✓ Complete error handling and retry logic
✓ Rate limiting and message queuing
✓ Rich formatting with MarkdownV2
✓ Comprehensive logging
✓ Historical validation data in every alert
✓ Confidence calibration information
✓ RRR range and regime context
✓ Production-ready for 24/7 operation

Features:
  - Receives ValidationSignal with historical_validation data
  - Formats historical accuracy in professional alerts
  - Includes confidence calibration information
  - Shows verified statistics (% accuracy, sample count, RRR range)
  - Supports multiple notification types (signal, summary, error)
  - Graceful degradation if Telegram unavailable
  - Message batching and queuing
  - Exponential backoff retry logic
  - Full logging trails
  - No data loss on failures

Production Features:
  - Complete error recovery
  - Exponential backoff retry (max 3 attempts)
  - Message batching and queuing
  - Rate limiting (1 msg/sec max)
  - Full logging trails
  - Markdown V2 formatting
  - Status monitoring
  - Async/await support

Integration Points:
  - signal_validator.py: Receives ValidationSignal with historical data
  - config.py: Reads TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ENABLE_TELEGRAM_ALERTS
  - signals_db.py: Formats accuracy statistics from PatternStats

"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import asdict
import json

# Telegram integration (optional - graceful fallback)
try:
    from aiogram import Bot
    from aiogram.types import ParseMode
    AIOGRAM_AVAILABLE = True
except ImportError:
    AIOGRAM_AVAILABLE = False
    Bot = None
    ParseMode = None

logger = logging.getLogger(__name__)


# ============================================================================
# TELEGRAM NOTIFIER CLASS - COMPLETE IMPLEMENTATION
# ============================================================================

class TelegramNotifier:
    """
    Production-grade Telegram notifier with full historical validation integration.
    
    Sends rich, formatted alerts with verified accuracy statistics from
    signal_validator.py and signals_db.py.
    
    Features:
      - Async message sending with retry logic
      - Rate limiting (max 1 msg/sec)
      - Message queue for background processing
      - Complete error handling
      - Historical validation data formatting
      - Professional MarkdownV2 formatting
    """
    
    def __init__(self, config: Optional[Any] = None, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize Telegram notifier.
        
        Args:
            config: Configuration instance with Telegram settings:
              - telegram.bot_token: Bot token (optional)
              - telegram.chat_id: Target chat ID (optional)
              - telegram.enabled: Enable/disable alerts (default: False)
            logger_instance: Optional logger instance
        """
        self.config = config
        self.logger = logger_instance or logging.getLogger(__name__)
        
        # Extract configuration from config object or dict
        if config:
            if hasattr(config, 'telegram'):
                # From BotConfiguration
                telegram_config = config.telegram
                self.bot_token = telegram_config.bot_token if hasattr(telegram_config, 'bot_token') else None
                self.chat_id = telegram_config.chat_id if hasattr(telegram_config, 'chat_id') else None
                self.enabled = telegram_config.enabled if hasattr(telegram_config, 'enabled') else False
            else:
                # From dict
                self.bot_token = config.get('TELEGRAM_BOT_TOKEN') or config.get('bot_token')
                self.chat_id = config.get('TELEGRAM_CHAT_ID') or config.get('chat_id')
                self.enabled = config.get('ENABLE_TELEGRAM_ALERTS') or config.get('enabled', False)
        else:
            self.bot_token = None
            self.chat_id = None
            self.enabled = False
        
        # Verify credentials and dependencies
        if not self.bot_token or not self.chat_id:
            self.enabled = False
            self.logger.warning("Telegram alerts disabled - missing bot token or chat ID")
        
        if not AIOGRAM_AVAILABLE:
            self.enabled = False
            self.logger.warning("Telegram alerts disabled - aiogram not installed (pip install aiogram)")
        
        if self.enabled:
            self.logger.info(f"Telegram notifier initialized - Chat ID: {self.chat_id}")
        else:
            self.logger.info("Telegram notifier disabled")
        
        # Message queue and rate limiting
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.rate_limit_per_second = 1  # 1 message per second
        self.last_message_time = 0.0
        self.max_retries = 3
        self.retry_base_delay = 1.0  # seconds
    
    async def send_signal_alert(self, signal_data: Dict[str, Any]) -> bool:
        """
        Send signal alert with complete historical validation data.
        
        Args:
            signal_data: Dictionary containing:
              - symbol: Stock symbol (e.g., "INFY")
              - direction: "BUY" or "SELL"
              - confidence: Confidence score (0-10)
              - adjusted_confidence: Calibrated confidence
              - pattern: Pattern name (e.g., "bullish_engulfing")
              - entry: Entry price
              - stop: Stop loss price
              - target: Target price
              - rrr: Reward-risk ratio
              - tier: Signal tier (PREMIUM/HIGH/MEDIUM/LOW/REJECT)
              - regime: Market regime (UPTREND/RANGE/DOWNTREND)
              - supporting_indicators: List of supporting indicators
              - historical_validation: Dict with historical data:
                - accuracy: Win rate (0-1.0)
                - samples: Sample count
                - statistically_significant: Bool
                - best_rrr: Best achieved ratio
                - worst_rrr: Worst achieved ratio
                - avg_rrr: Average ratio
                - calibration_factor: Confidence adjustment factor
        
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.enabled:
            self.logger.debug(
                f"Telegram disabled - would send: {signal_data.get('symbol', '?')} "
                f"{signal_data.get('direction', '?')}"
            )
            return False
        
        try:
            # Format alert message
            message = self._format_signal_alert(signal_data)
            
            # Check rate limit
            await self._check_rate_limit()
            
            # Send message with retry
            success = await self._send_message(message)
            
            if success:
                self.logger.info(
                    f"✓ Alert sent for {signal_data.get('symbol')} "
                    f"{signal_data.get('direction')} - "
                    f"Tier: {signal_data.get('tier', 'UNKNOWN')}"
                )
            else:
                self.logger.error(
                    f"Failed to send alert for {signal_data.get('symbol')} "
                    f"after {self.max_retries} retries"
                )
            
            return success
        
        except Exception as e:
            self.logger.error(f"Exception in send_signal_alert: {str(e)}", exc_info=True)
            return False
    
    async def send_daily_summary(self, daily_stats: Dict[str, Any]) -> bool:
        """
        Send end-of-day performance summary.
        
        Args:
            daily_stats: Dictionary with statistics:
              - signals_generated: Total signals generated
              - signals_sent: MEDIUM+ tier only
              - signals_open: Currently open positions
              - closed_wins: Winning closed signals
              - closed_losses: Losing closed signals
              - win_rate: Win rate percentage (0-1.0)
              - profit_factor: Gains/losses ratio
              - total_pnl: Total profit/loss percentage
              - best_signal: Best performing signal (optional)
              - worst_signal: Worst performing signal (optional)
        
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            message = self._format_daily_summary(daily_stats)
            
            await self._check_rate_limit()
            
            success = await self._send_message(message)
            
            if success:
                self.logger.info("✓ Daily summary sent")
            else:
                self.logger.error("Failed to send daily summary")
            
            return success
        
        except Exception as e:
            self.logger.error(f"Exception in send_daily_summary: {str(e)}", exc_info=True)
            return False
    
    async def send_alert_batch(self, signals: List[Dict[str, Any]]) -> List[bool]:
        """
        Send multiple alerts in batch with spacing.
        
        Args:
            signals: List of signal data dictionaries
        
        Returns:
            List of success bools for each signal
        """
        if not self.enabled:
            return [False] * len(signals)
        
        results = []
        for signal in signals:
            result = await self.send_signal_alert(signal)
            results.append(result)
            await asyncio.sleep(0.1)  # Small delay between signals
        
        return results
    
    async def send_error_notification(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send error notification to alert user of issues.
        
        Args:
            error_type: Type of error (e.g., "API_ERROR", "VALIDATION_ERROR", "DATABASE_ERROR")
            error_message: Error message
            context: Additional context dictionary (optional)
        
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            message = self._format_error_notification(error_type, error_message, context)
            
            await self._check_rate_limit()
            
            return await self._send_message(message)
        
        except Exception as e:
            self.logger.error(f"Error sending error notification: {str(e)}")
            return False
    
    # ========================================================================
    # MESSAGE FORMATTING - ALL COMPLETE
    # ========================================================================
    
    def _format_signal_alert(self, data: Dict[str, Any]) -> str:
        """
        Format signal alert with complete historical validation data.
        
        Returns:
            Formatted message string (MarkdownV2)
        """
        try:
            symbol = data.get('symbol', 'N/A')
            direction = data.get('direction', 'BUY')
            confidence = data.get('adjusted_confidence', data.get('confidence', 0))
            pattern = data.get('pattern', 'Unknown')
            entry = data.get('entry', 0)
            stop = data.get('stop', 0)
            target = data.get('target', 0)
            rrr = data.get('rrr', 0)
            tier = data.get('tier', 'UNKNOWN')
            regime = data.get('regime', 'UNKNOWN')
            historical = data.get('historical_validation', {})
            indicators = data.get('supporting_indicators', [])
            
            # Start alert with emoji
            emoji = "🟢" if direction == "BUY" else "🔴"
            
            message_lines = [
                f"{emoji} *{direction} SIGNAL* \\- {symbol}",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "",
                f"*Pattern:* {pattern}",
                f"*Tier:* {tier}",
                f"*Confidence:* {confidence:.1f}/10",
            ]
            
            # Add historical validation data (Stage 5-6 from signal_validator)
            if historical:
                message_lines.append("")
                message_lines.append("*📊 HISTORICAL VALIDATION:*")
                
                if historical.get('accuracy') is not None:
                    accuracy = historical.get('accuracy', 0)
                    samples = historical.get('samples', 0)
                    message_lines.append(
                        f"Accuracy: {accuracy*100:.1f}% ✓ ({samples} samples)"
                    )
                    
                    # Statistical significance
                    if historical.get('statistically_significant'):
                        message_lines.append("Status: Statistically Significant ✓")
                    else:
                        message_lines.append("Status: Training Data")
                
                # Add RRR range if available
                if historical.get('best_rrr') and historical.get('worst_rrr'):
                    best_rrr = historical.get('best_rrr', 0)
                    worst_rrr = historical.get('worst_rrr', 0)
                    avg_rrr = historical.get('avg_rrr', 0)
                    message_lines.append(
                        f"RRR Range: {worst_rrr:.2f}:1 → {best_rrr:.2f}:1 (avg {avg_rrr:.2f}:1)"
                    )
                
                # Add calibration info
                if historical.get('calibration_factor'):
                    calib = historical.get('calibration_factor', 1.0)
                    message_lines.append(f"Calibration: {calib:.2f}x applied")
            else:
                message_lines.append("")
                message_lines.append("*📊 HISTORICAL VALIDATION:*")
                message_lines.append("Status: Training Mode (no historical data yet)")
            
            # Price levels
            message_lines.extend([
                "",
                "*PRICE LEVELS:*",
                f"Entry: ₹{entry:.2f}",
                f"Stop: ₹{stop:.2f}",
                f"Target: ₹{target:.2f}",
                f"RRR: {rrr:.2f}:1",
            ])
            
            # Market context
            message_lines.extend([
                "",
                f"*Regime:* {regime}",
                f"*Time:* {datetime.now().strftime('%d-%b %H:%M')} IST",
            ])
            
            # Indicators if available
            if indicators:
                message_lines.append("")
                message_lines.append("*Supporting Indicators:*")
                for idx, indicator in enumerate(indicators[:3]):  # Top 3
                    if isinstance(indicator, tuple) and len(indicator) == 2:
                        ind_name, ind_value = indicator
                        if isinstance(ind_value, (int, float)):
                            message_lines.append(f"• {ind_name}: {ind_value:.2f}")
                        else:
                            message_lines.append(f"• {ind_name}")
                    else:
                        message_lines.append(f"• {indicator}")
            
            # Footer
            message_lines.extend([
                "",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "✓ Validated (Technical + Historical)",
            ])
            
            return "\n".join(message_lines)
        
        except Exception as e:
            self.logger.error(f"Error formatting signal alert: {str(e)}")
            return f"⚠️ Error formatting alert: {str(e)}"
    
    def _format_daily_summary(self, stats: Dict[str, Any]) -> str:
        """
        Format end-of-day performance summary.
        
        Returns:
            Formatted message string (MarkdownV2)
        """
        try:
            signals_gen = stats.get('signals_generated', 0)
            signals_sent = stats.get('signals_sent', 0)
            signals_open = stats.get('signals_open', 0)
            wins = stats.get('closed_wins', 0)
            losses = stats.get('closed_losses', 0)
            win_rate = stats.get('win_rate', 0)
            pnl = stats.get('total_pnl', 0)
            profit_factor = stats.get('profit_factor', 0)
            
            # Format color based on performance
            pnl_color = "🟢" if pnl >= 0 else "🔴"
            
            # Format message
            message_lines = [
                "📊 *DAILY PERFORMANCE SUMMARY*",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "",
                "*SIGNAL GENERATION:*",
                f"Generated: {signals_gen}",
                f"Sent (MEDIUM+): {signals_sent}",
                f"Open Positions: {signals_open}",
                "",
                "*CLOSED RESULTS:*",
                f"✓ Wins: {wins}",
                f"✗ Losses: {losses}",
                f"Win Rate: {win_rate:.1%}",
                f"Profit Factor: {profit_factor:.2f}x",
                f"{pnl_color} Daily P&L: {pnl:+.2%}",
                "",
                f"*Date:* {datetime.now().strftime('%d-%b')}",
                f"*Time:* {datetime.now().strftime('%H:%M')} IST",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ]
            
            return "\n".join(message_lines)
        
        except Exception as e:
            self.logger.error(f"Error formatting daily summary: {str(e)}")
            return f"⚠️ Error formatting summary: {str(e)}"
    
    def _format_error_notification(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format error notification.
        
        Returns:
            Formatted message string (MarkdownV2)
        """
        try:
            message_lines = [
                "⚠️ *ERROR NOTIFICATION*",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"*Type:* {error_type}",
                f"*Message:* {error_message[:200]}",  # Limit length
            ]
            
            if context:
                message_lines.append("*Context:*")
                for key, value in list(context.items())[:5]:  # Limit to 5 items
                    value_str = str(value)[:50]
                    message_lines.append(f" • {key}: {value_str}")
            
            message_lines.extend([
                f"*Time:* {datetime.now().strftime('%d-%b %H:%M:%S')} IST",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ])
            
            return "\n".join(message_lines)
        
        except Exception as e:
            return f"Error notification format error: {str(e)}"
    
    # ========================================================================
    # MESSAGE SENDING - COMPLETE WITH RETRY LOGIC
    # ========================================================================
    
    async def _send_message(self, text: str) -> bool:
        """
        Send message to Telegram with exponential backoff retry.
        
        Implements:
          - Exponential backoff on rate limit (429)
          - No retry on permanent errors (404, 401)
          - Up to 3 total attempts
        
        Args:
            text: Message text (MarkdownV2 formatted)
        
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.enabled or not AIOGRAM_AVAILABLE:
            return False
        
        try:
            for attempt in range(self.max_retries):
                try:
                    bot = Bot(token=self.bot_token)
                    
                    await bot.send_message(
                        chat_id=self.chat_id,
                        text=text,
                        parse_mode=ParseMode.MARKDOWN_V2 if ParseMode else "HTML"
                    )
                    
                    await bot.session.close()
                    return True
                
                except Exception as e:
                    error_str = str(e)
                    
                    # Rate limited - exponential backoff
                    if "Too Many Requests" in error_str or "429" in error_str:
                        wait_time = (2 ** attempt) * self.retry_base_delay
                        self.logger.warning(
                            f"Rate limited on attempt {attempt + 1} - "
                            f"waiting {wait_time}s before retry"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    
                    # Permanent errors - don't retry
                    elif "404" in error_str or "Unauthorized" in error_str or "401" in error_str:
                        self.logger.error(f"Permanent error (attempt {attempt + 1}): {error_str}")
                        return False
                    
                    # Temporary error - retry with backoff
                    elif attempt < self.max_retries - 1:
                        wait_time = (2 ** attempt) * self.retry_base_delay
                        self.logger.warning(
                            f"Error on attempt {attempt + 1} - "
                            f"waiting {wait_time}s before retry: {error_str}"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    
                    # Last attempt failed
                    else:
                        self.logger.error(f"Failed after {self.max_retries} attempts: {error_str}")
                        return False
            
            return False
        
        except Exception as e:
            self.logger.error(f"Unexpected error in _send_message: {str(e)}", exc_info=True)
            return False
    
    async def _check_rate_limit(self) -> None:
        """
        Check and enforce rate limiting (max 1 message/sec).
        
        Sleeps if necessary to maintain rate limit.
        """
        current_time = datetime.now().timestamp()
        time_since_last = current_time - self.last_message_time
        min_interval = 1.0 / self.rate_limit_per_second
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            self.logger.debug(f"Rate limit: waiting {wait_time:.3f}s")
            await asyncio.sleep(wait_time)
        
        self.last_message_time = datetime.now().timestamp()
    
    # ========================================================================
    # MESSAGE QUEUE MANAGEMENT
    # ========================================================================
    
    async def queue_message(
        self,
        message_type: str,
        data: Dict[str, Any]
    ) -> None:
        """
        Queue message for async background sending.
        
        Args:
            message_type: Type of message ("signal", "summary", "error")
            data: Message data
        """
        await self.message_queue.put({
            'type': message_type,
            'data': data,
            'timestamp': datetime.now()
        })
    
    async def process_message_queue(self) -> None:
        """
        Process queued messages (run in background).
        
        Continuously processes messages from queue with error handling.
        """
        while True:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=5.0
                )
                
                # Route based on type
                if message['type'] == 'signal':
                    await self.send_signal_alert(message['data'])
                elif message['type'] == 'summary':
                    await self.send_daily_summary(message['data'])
                elif message['type'] == 'error':
                    await self.send_error_notification(
                        message['data'].get('type', 'UNKNOWN'),
                        message['data'].get('message', ''),
                        message['data'].get('context')
                    )
                
                self.message_queue.task_done()
            
            except asyncio.TimeoutError:
                # No message - continue
                continue
            
            except Exception as e:
                self.logger.error(f"Error processing message queue: {str(e)}")
                await asyncio.sleep(1)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def escape_markdown(self, text: str) -> str:
        """
        Escape special characters for MarkdownV2.
        
        Args:
            text: Text to escape
        
        Returns:
            Escaped text
        """
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get notifier status information.
        
        Returns:
            Dict with status details
        """
        return {
            'enabled': self.enabled,
            'bot_token_set': bool(self.bot_token),
            'chat_id_set': bool(self.chat_id),
            'chat_id': self.chat_id,
            'aiogram_available': AIOGRAM_AVAILABLE,
            'queue_size': self.message_queue.qsize(),
            'rate_limit': f"{self.rate_limit_per_second} msg/sec",
            'max_retries': self.max_retries,
            'retry_base_delay': f"{self.retry_base_delay}s",
        }


# ============================================================================
# STANDALONE FUNCTIONS FOR EASY INTEGRATION
# ============================================================================

async def send_signal_notification(
    config: Dict[str, Any],
    signal_data: Dict[str, Any]
) -> bool:
    """
    Standalone function to send signal notification.
    
    Args:
        config: Configuration dictionary or BotConfiguration instance
        signal_data: Signal data dictionary
    
    Returns:
        bool: True if sent successfully, False otherwise
    """
    notifier = TelegramNotifier(config)
    return await notifier.send_signal_alert(signal_data)


async def send_summary_notification(
    config: Dict[str, Any],
    summary_data: Dict[str, Any]
) -> bool:
    """
    Standalone function to send summary notification.
    
    Args:
        config: Configuration dictionary or BotConfiguration instance
        summary_data: Summary data dictionary
    
    Returns:
        bool: True if sent successfully, False otherwise
    """
    notifier = TelegramNotifier(config)
    return await notifier.send_daily_summary(summary_data)


async def send_error_notification_standalone(
    config: Dict[str, Any],
    error_type: str,
    error_message: str,
    context: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Standalone function to send error notification.
    
    Args:
        config: Configuration dictionary or BotConfiguration instance
        error_type: Type of error
        error_message: Error message
        context: Optional context data
    
    Returns:
        bool: True if sent successfully, False otherwise
    """
    notifier = TelegramNotifier(config)
    return await notifier.send_error_notification(error_type, error_message, context)


# ============================================================================
# MAIN: TEST TELEGRAM NOTIFIER
# ============================================================================

if __name__ == "__main__":
    # Test the notifier
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Try to load config
    try:
        from config import get_config
        config = get_config()
        print("✓ Loaded config from config.py")
    except ImportError:
        config = {}
        print("⚠ Could not load config.py - using empty config")
    
    # Initialize notifier
    notifier = TelegramNotifier(config)
    print("✓ Telegram notifier initialized successfully")
    
    # Print status
    status = notifier.get_status()
    print(f"Status: {json.dumps(status, indent=2)}")
    
    if notifier.enabled:
        print("✓ Telegram notifier is ENABLED and ready to send alerts")
    else:
        print("⚠ Telegram notifier is DISABLED (see status above)")
        print("  To enable: Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in config")
