"""
TELEGRAM NOTIFIER - SIGNAL NOTIFICATION & ALERT SYSTEM
======================================================

This module handles:
✓ Rich Telegram message formatting
✓ Signal notifications with full analysis
✓ Error & warning alerts
✓ Daily performance summaries
✓ Adhoc signal validation notifications
✓ Rate limiting and retry logic
✓ Message queuing for reliability

Author: rahulreddyallu
Version: 4.0.0 (Institutional Grade)
Date: 2025-11-30
"""

import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass
import textwrap

try:
    from aiogram import Bot, Dispatcher
    from aiogram.types import Message
    from aiogram.exceptions import TelegramAPIError
except ImportError:
    logging.error("aiogram not installed. Install with: pip install aiogram")


# ============================================================================
# MESSAGE TEMPLATES
# ============================================================================

class MessageTemplates:
    """Pre-formatted message templates for different scenarios"""
    
    @staticmethod
    def escape_markdown(text: str) -> str:
        """Escape special characters for Telegram MarkdownV2"""
        if not text:
            return "N/A"
        
        special_chars = r'_*[]()~`>#+-=|{}.!'
        escaped = re.sub(f'([{re.escape(special_chars)}])', r'\\\1', str(text))
        return escaped
    
    @staticmethod
    def signal_alert(
        symbol: str,
        direction: str,
        tier: str,
        confidence: int,
        pattern: str,
        entry: float,
        stop: float,
        target: float,
        rrr: float,
        win_rate: float,
        indicators: List[str],
        regime: str
    ) -> str:
        """
        Format a signal alert message
        
        Returns:
            Formatted Telegram message
        """
        
        # Signal emoji and color
        if direction == "BUY":
            emoji = "🟢"
            type_text = "BUY SIGNAL"
        else:
            emoji = "🔴"
            type_text = "SELL SIGNAL"
        
        # Tier emoji
        tier_emojis = {
            "PREMIUM": "⭐⭐⭐",
            "HIGH": "⭐⭐",
            "MEDIUM": "⭐",
            "LOW": "⚠️"
        }
        
        tier_emoji = tier_emojis.get(tier, "❓")
        
        # Build message
        lines = [
            f"{emoji} *{type_text}*",
            f"",
            f"*Symbol:* `{MessageTemplates.escape_markdown(symbol)}`",
            f"*Pattern:* {MessageTemplates.escape_markdown(pattern)}",
            f"*Confidence:* {confidence}/10 {tier_emoji}",
            f"*Win Rate:* {win_rate*100:.0f}%",
            f"",
            f"*Entry:* ₹{entry:.2f}",
            f"*Stop Loss:* ₹{stop:.2f}",
            f"*Target:* ₹{target:.2f}",
            f"*RRR:* {rrr:.2f}:1 {'✅' if rrr >= 1.5 else '⚠️'}",
            f"",
            f"*Market Regime:* {MessageTemplates.escape_markdown(regime)}",
            f"*Confirming Indicators:* {', '.join(indicators)}",
            f"",
            f"*⏰ Time:* {datetime.now().strftime('%H:%M:%S IST')}"
        ]
        
        return "\n".join(lines)
    
    @staticmethod
    def validation_details(
        validation_signal: Dict[str, Any]
    ) -> str:
        """Format detailed validation breakdown"""
        
        lines = [
            f"*Validation Breakdown:*",
            f"",
            f"*Pattern Score:* {validation_signal.get('pattern_score', 0)}/3 ({validation_signal.get('patterns', [])[:1]})",
            f"*Indicator Score:* {validation_signal.get('indicator_score', 0)}/3",
            f"*Context Score:* {validation_signal.get('context_score', 0)}/2",
            f"*Risk Score:* {validation_signal.get('risk_score', 0)}/2",
            f"",
            f"*Total Score:* {validation_signal.get('total_score', 0)}/10",
        ]
        
        return "\n".join(lines)
    
    @staticmethod
    def error_alert(error_type: str, symbol: str, error_msg: str) -> str:
        """Format error notification"""
        
        lines = [
            f"🚨 *ERROR ALERT*",
            f"",
            f"*Type:* {MessageTemplates.escape_markdown(error_type)}",
            f"*Symbol:* `{MessageTemplates.escape_markdown(symbol)}`",
            f"*Error:* {MessageTemplates.escape_markdown(error_msg[:100])}",
            f"",
            f"*Time:* {datetime.now().strftime('%H:%M:%S IST')}"
        ]
        
        return "\n".join(lines)
    
    @staticmethod
    def daily_summary(
        signals_generated: int,
        signals_sent: int,
        avg_confidence: float,
        best_pattern: str,
        win_rate: float,
        profit_factor: float
    ) -> str:
        """Format daily summary"""
        
        lines = [
            f"📊 *DAILY SUMMARY*",
            f"",
            f"*Signals Generated:* {signals_generated}",
            f"*Signals Sent:* {signals_sent}",
            f"*Avg Confidence:* {avg_confidence:.1f}/10",
            f"",
            f"*Best Pattern:* {best_pattern}",
            f"*Win Rate:* {win_rate*100:.1f}%",
            f"*Profit Factor:* {profit_factor:.2f}x",
            f"",
            f"*Generated:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}"
        ]
        
        return "\n".join(lines)
    
    @staticmethod
    def adhoc_validation(symbol: str, status: str) -> str:
        """Format adhoc signal validation request"""
        
        emoji = "✅" if status == "ANALYZING" else "⚠️"
        
        lines = [
            f"{emoji} *ADHOC SIGNAL VALIDATION*",
            f"",
            f"*Symbol:* `{MessageTemplates.escape_markdown(symbol)}`",
            f"*Status:* {status}",
            f"",
            f"*Time:* {datetime.now().strftime('%H:%M:%S IST')}"
        ]
        
        return "\n".join(lines)


# ============================================================================
# TELEGRAM NOTIFIER CLASS
# ============================================================================

@dataclass
class NotificationQueue:
    """Queue for storing notifications"""
    message: str
    priority: int  # 1-5 (5 = highest priority)
    retry_count: int = 0
    max_retries: int = 3
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TelegramNotifier:
    """
    Telegram notification system with:
    - Rate limiting
    - Retry logic
    - Message queuing
    - Error handling
    """
    
    def __init__(self, bot_token: str, chat_id: str, logger: Optional[logging.Logger] = None):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram bot token
            chat_id: Target chat ID
            logger: Optional logger
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.logger = logger or logging.getLogger(__name__)
        
        self.bot = None
        self.dispatcher = None
        self.message_queue: List[NotificationQueue] = []
        self.last_message_time = 0
        self.min_interval = 1  # Min 1 second between messages
        self.rate_limit_reset = 0
    
    async def initialize(self) -> bool:
        """Initialize bot connection"""
        try:
            self.bot = Bot(token=self.bot_token)
            self.dispatcher = Dispatcher()
            
            # Test connection
            bot_info = await self.bot.get_me()
            self.logger.info(f"✅ Telegram bot connected: @{bot_info.username}")
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Telegram bot: {str(e)}")
            return False
    
    async def shutdown(self):
        """Cleanup bot connection"""
        if self.bot:
            await self.bot.session.close()
    
    async def send_signal_alert(
        self,
        symbol: str,
        direction: str,
        tier: str,
        confidence: int,
        pattern: str,
        entry: float,
        stop: float,
        target: float,
        rrr: float,
        win_rate: float,
        indicators: List[str],
        regime: str
    ) -> bool:
        """
        Send a signal alert notification
        
        Args:
            All signal parameters
        
        Returns:
            True if sent successfully
        """
        
        message = MessageTemplates.signal_alert(
            symbol=symbol,
            direction=direction,
            tier=tier,
            confidence=confidence,
            pattern=pattern,
            entry=entry,
            stop=stop,
            target=target,
            rrr=rrr,
            win_rate=win_rate,
            indicators=indicators,
            regime=regime
        )
        
        return await self._send_message(message, priority=4)
    
    async def send_error_alert(
        self,
        error_type: str,
        symbol: str,
        error_msg: str
    ) -> bool:
        """Send error notification"""
        
        message = MessageTemplates.error_alert(
            error_type=error_type,
            symbol=symbol,
            error_msg=error_msg
        )
        
        return await self._send_message(message, priority=5)  # Highest priority
    
    async def send_daily_summary(
        self,
        signals_generated: int,
        signals_sent: int,
        avg_confidence: float,
        best_pattern: str,
        win_rate: float,
        profit_factor: float
    ) -> bool:
        """Send daily summary"""
        
        message = MessageTemplates.daily_summary(
            signals_generated=signals_generated,
            signals_sent=signals_sent,
            avg_confidence=avg_confidence,
            best_pattern=best_pattern,
            win_rate=win_rate,
            profit_factor=profit_factor
        )
        
        return await self._send_message(message, priority=2)
    
    async def send_adhoc_notification(self, symbol: str, status: str) -> bool:
        """Send adhoc validation notification"""
        
        message = MessageTemplates.adhoc_validation(symbol=symbol, status=status)
        return await self._send_message(message, priority=3)
    
    async def _send_message(
        self,
        message: str,
        priority: int = 2,
        max_retries: int = 3
    ) -> bool:
        """
        Send message with retry logic and rate limiting
        
        Args:
            message: Message text
            priority: Message priority (1-5)
            max_retries: Number of retry attempts
        
        Returns:
            True if sent successfully
        """
        
        if not self.bot:
            self.logger.warning("Telegram bot not initialized")
            return False
        
        # Add to queue
        queue_item = NotificationQueue(
            message=message,
            priority=priority,
            max_retries=max_retries
        )
        self.message_queue.append(queue_item)
        
        # Sort by priority (highest first)
        self.message_queue.sort(key=lambda x: x.priority, reverse=True)
        
        # Process queue
        return await self._process_queue()
    
    async def _process_queue(self) -> bool:
        """Process message queue"""
        
        while self.message_queue:
            # Apply rate limiting
            await self._apply_rate_limit()
            
            item = self.message_queue.pop(0)
            
            try:
                # Chunk message if too long (Telegram limit: 4096 characters)
                messages = self._chunk_message(item.message)
                
                for msg in messages:
                    await self.bot.send_message(
                        chat_id=self.chat_id,
                        text=msg,
                        parse_mode="MarkdownV2"
                    )
                    
                    # Update rate limit
                    self.last_message_time = asyncio.get_event_loop().time()
                
                self.logger.debug(f"✅ Message sent (priority={item.priority})")
                return True
            
            except TelegramAPIError as e:
                if "Too Many Requests" in str(e):
                    # Extract retry_after from error
                    retry_after = self._extract_retry_after(str(e))
                    self.rate_limit_reset = asyncio.get_event_loop().time() + retry_after
                    
                    # Re-queue the message
                    if item.retry_count < item.max_retries:
                        item.retry_count += 1
                        self.message_queue.insert(0, item)
                        self.logger.warning(f"Rate limited. Retry in {retry_after}s")
                        await asyncio.sleep(retry_after)
                        return await self._process_queue()
                    else:
                        self.logger.error(f"Max retries exceeded for message")
                        return False
                else:
                    self.logger.error(f"Telegram error: {str(e)}")
                    return False
            
            except Exception as e:
                self.logger.error(f"Unexpected error sending message: {str(e)}")
                return False
        
        return True
    
    async def _apply_rate_limit(self):
        """Apply rate limiting between messages"""
        
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_message_time
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        # Check global rate limit
        if current_time < self.rate_limit_reset:
            wait_time = self.rate_limit_reset - current_time
            self.logger.warning(f"Global rate limit. Waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
    
    @staticmethod
    def _chunk_message(message: str, max_length: int = 4090) -> List[str]:
        """
        Split message into chunks if too long
        
        Args:
            message: Message text
            max_length: Maximum length per chunk
        
        Returns:
            List of message chunks
        """
        
        if len(message) <= max_length:
            return [message]
        
        chunks = []
        current_chunk = ""
        
        for line in message.split("\n"):
            if len(current_chunk) + len(line) + 1 > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line
            else:
                if current_chunk:
                    current_chunk += "\n" + line
                else:
                    current_chunk = line
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    @staticmethod
    def _extract_retry_after(error_msg: str) -> int:
        """Extract retry_after seconds from Telegram error"""
        
        match = re.search(r"retry after (\d+)", error_msg)
        if match:
            return int(match.group(1))
        
        return 30  # Default to 30 seconds


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def test_telegram_connection(bot_token: str, chat_id: str) -> bool:
    """Test Telegram bot connection"""
    
    try:
        bot = Bot(token=bot_token)
        
        test_msg = "🧪 *Test Message*\n\nTelegram connection test successful\\!"
        await bot.send_message(
            chat_id=chat_id,
            text=test_msg,
            parse_mode="MarkdownV2"
        )
        
        await bot.session.close()
        return True
    
    except Exception as e:
        logging.error(f"Telegram connection test failed: {str(e)}")
        return False


# ============================================================================
# SIGNAL FORMATTER
# ============================================================================

def format_validation_signal_for_telegram(validation_signal: Dict[str, Any]) -> str:
    """
    Convert ValidationSignal to Telegram message format
    
    Args:
        validation_signal: Dictionary representation of ValidationSignal
    
    Returns:
        Formatted Telegram message
    """
    
    symbol = validation_signal.get('symbol', 'UNKNOWN')
    direction = validation_signal.get('direction', 'UNKNOWN')
    tier = validation_signal.get('tier', 'UNKNOWN')
    confidence = validation_signal.get('confidence', 0)
    patterns = validation_signal.get('patterns', [])
    supporting = validation_signal.get('supporting', [])
    win_rate = validation_signal.get('win_rate', '0%')
    
    emoji = "🟢" if direction == "BUY" else "🔴"
    
    message = f"""{emoji} *{direction} SIGNAL - {symbol}*

*Tier:* {tier} ({confidence}/10)
*Pattern:* {', '.join(patterns) if patterns else 'Multiple'}
*Win Rate:* {win_rate}

*Confirmations:*
{', '.join(['✅ ' + ind for ind in supporting]) if supporting else 'Loading...'}

*Status:* {'VALIDATED ✅' if validation_signal.get('passed') else 'REJECTED ❌'}

_Time: {datetime.now().strftime('%H:%M:%S IST')}_
"""
    
    return message


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test message templates
    msg = MessageTemplates.signal_alert(
        symbol="INFY",
        direction="BUY",
        tier="HIGH",
        confidence=8,
        pattern="Bullish Engulfing",
        entry=1650.50,
        stop=1640.00,
        target=1680.00,
        rrr=2.0,
        win_rate=0.65,
        indicators=["RSI", "MACD", "Volume"],
        regime="UPTREND"
    )
    print(msg)
