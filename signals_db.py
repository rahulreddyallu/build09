# signals_db.py - COMPLETE PRODUCTION VERSION (v3.2.0)
# ==================================================================================
# Historical Pattern Accuracy Database with Full Integration
# Stores and manages pattern accuracy data by market regime
# Used for pre-alert validation and confidence calibration in signal_validator.py
# ==================================================================================
#
# Author: rahulreddyallu
# Version: 3.2.0 (Production - Fully Integrated)
# Date: 2025-12-01
#
# ==================================================================================

"""
SIGNALS DATABASE - HISTORICAL PATTERN ACCURACY TRACKING

===================================================================================

This module implements a COMPLETE historical accuracy tracking system for
candlestick patterns and technical indicators, fully integrated with
signal_validator.py v4.5.1 and config.py v4.1.0:

✓ MarketRegime enum - 7 market classifications
✓ PatternStats dataclass - Per-pattern, per-regime statistics
✓ ConfidenceCalibration - Maps confidence to expected accuracy
✓ PatternAccuracyDatabase - In-memory storage and management
✓ Export/Import - JSON persistence
✓ Summary Reports - Pattern analysis and reporting

Production Features:
  - Complete error handling on all paths
  - Comprehensive logging at every operation
  - JSON serialization/deserialization
  - Memory-efficient in-memory storage
  - Regime-specific accuracy tracking
  - Statistical significance validation
  - Confidence adjustment based on historical data
  - Seamless integration with signal_validator.py

Market Regimes Tracked:
  - STRONG_UPTREND (ADX > 40, +DI > -DI)
  - UPTREND (ADX 25-40, +DI > -DI)
  - WEAK_UPTREND (ADX < 25, +DI > -DI)
  - RANGE (ADX < 25, equal DI)
  - WEAK_DOWNTREND (ADX < 25, -DI > +DI)
  - DOWNTREND (ADX 25-40, -DI > +DI)
  - STRONG_DOWNTREND (ADX > 40, -DI > +DI)

Integration Points:
  - signal_validator.py: Queries for pattern accuracy
  - backtest_report.py: Records results from trades
  - config.py: Uses thresholds from config

"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# MARKET REGIME CLASSIFICATION
# ============================================================================

class MarketRegime(Enum):
    """Market regime classification (7 levels)"""
    STRONG_UPTREND = "STRONG_UPTREND"      # ADX > 40, +DI > -DI
    UPTREND = "UPTREND"                    # ADX 25-40, +DI > -DI
    WEAK_UPTREND = "WEAK_UPTREND"          # ADX < 25, +DI > -DI
    RANGE = "RANGE"                        # ADX < 25, equal DI
    WEAK_DOWNTREND = "WEAK_DOWNTREND"      # ADX < 25, -DI > +DI
    DOWNTREND = "DOWNTREND"                # ADX 25-40, -DI > +DI
    STRONG_DOWNTREND = "STRONG_DOWNTREND"  # ADX > 40, -DI > +DI


# ============================================================================
# PATTERN STATISTICS DATACLASS
# ============================================================================

@dataclass
class PatternStats:
    """Statistics for a pattern in a specific market regime"""
    pattern_name: str
    regime: MarketRegime
    
    # Performance metrics
    total_occurrences: int = 0
    winning_occurrences: int = 0
    losing_occurrences: int = 0
    
    # RRR tracking (all achieved ratios)
    rrr_values: List[float] = field(default_factory=list)
    
    # P&L tracking (all returns in %)
    pnl_values: List[float] = field(default_factory=list)
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    created_date: datetime = field(default_factory=datetime.now)
    
    def win_rate(self) -> float:
        """Calculate win rate (0-1.0)"""
        if self.total_occurrences == 0:
            return 0.0
        return self.winning_occurrences / self.total_occurrences
    
    def loss_rate(self) -> float:
        """Calculate loss rate (0-1.0)"""
        if self.total_occurrences == 0:
            return 0.0
        return self.losing_occurrences / self.total_occurrences
    
    def best_rrr(self) -> float:
        """Get best risk-reward ratio achieved"""
        return max(self.rrr_values) if self.rrr_values else 0.0
    
    def worst_rrr(self) -> float:
        """Get worst risk-reward ratio achieved"""
        return min(self.rrr_values) if self.rrr_values else 0.0
    
    def avg_rrr(self) -> float:
        """Calculate average RRR"""
        return float(np.mean(self.rrr_values)) if self.rrr_values else 0.0
    
    def median_rrr(self) -> float:
        """Calculate median RRR"""
        return float(np.median(self.rrr_values)) if self.rrr_values else 0.0
    
    def std_dev_rrr(self) -> float:
        """Calculate standard deviation of RRR"""
        return float(np.std(self.rrr_values)) if len(self.rrr_values) > 1 else 0.0
    
    def avg_pnl(self) -> float:
        """Calculate average P&L percentage"""
        return float(np.mean(self.pnl_values)) if self.pnl_values else 0.0
    
    def median_pnl(self) -> float:
        """Calculate median P&L percentage"""
        return float(np.median(self.pnl_values)) if self.pnl_values else 0.0
    
    def std_dev_pnl(self) -> float:
        """Calculate standard deviation of P&L"""
        return float(np.std(self.pnl_values)) if len(self.pnl_values) > 1 else 0.0
    
    def max_pnl(self) -> float:
        """Get maximum P&L achieved"""
        return float(max(self.pnl_values)) if self.pnl_values else 0.0
    
    def min_pnl(self) -> float:
        """Get minimum P&L achieved (largest loss)"""
        return float(min(self.pnl_values)) if self.pnl_values else 0.0
    
    def sample_size(self) -> int:
        """Get total number of samples"""
        return self.total_occurrences
    
    def is_statistically_significant(self, min_samples: int = 10) -> bool:
        """Check if sample size meets minimum threshold"""
        return self.sample_size() >= min_samples
    
    def expected_value(self) -> float:
        """Calculate expected value of pattern"""
        if not self.pnl_values:
            return 0.0
        win_rate = self.win_rate()
        loss_rate = self.loss_rate()
        avg_win = float(np.mean([p for p in self.pnl_values if p > 0])) if any(p > 0 for p in self.pnl_values) else 0
        avg_loss = float(np.mean([p for p in self.pnl_values if p < 0])) if any(p < 0 for p in self.pnl_values) else 0
        return (win_rate * avg_win) + (loss_rate * avg_loss)
    
    def profit_factor(self) -> float:
        """Calculate profit factor (total wins / total losses)"""
        wins = sum(p for p in self.pnl_values if p > 0)
        losses = abs(sum(p for p in self.pnl_values if p < 0))
        if losses == 0:
            return float('inf') if wins > 0 else 1.0
        return wins / losses
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'pattern_name': self.pattern_name,
            'regime': self.regime.value,
            'total_occurrences': self.total_occurrences,
            'winning_occurrences': self.winning_occurrences,
            'losing_occurrences': self.losing_occurrences,
            'win_rate': round(self.win_rate(), 4),
            'loss_rate': round(self.loss_rate(), 4),
            'best_rrr': round(self.best_rrr(), 3),
            'worst_rrr': round(self.worst_rrr(), 3),
            'avg_rrr': round(self.avg_rrr(), 3),
            'median_rrr': round(self.median_rrr(), 3),
            'std_dev_rrr': round(self.std_dev_rrr(), 3),
            'avg_pnl': round(self.avg_pnl(), 3),
            'median_pnl': round(self.median_pnl(), 3),
            'std_dev_pnl': round(self.std_dev_pnl(), 3),
            'max_pnl': round(self.max_pnl(), 3),
            'min_pnl': round(self.min_pnl(), 3),
            'sample_size': self.sample_size(),
            'statistically_significant': self.is_statistically_significant(),
            'expected_value': round(self.expected_value(), 3),
            'profit_factor': round(self.profit_factor(), 3),
            'last_updated': self.last_updated.isoformat(),
            'created_date': self.created_date.isoformat(),
        }


# ============================================================================
# CONFIDENCE CALIBRATION
# ============================================================================

@dataclass
class ConfidenceCalibration:
    """Maps confidence scores to expected accuracy"""
    confidence_ranges: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with default calibration ranges"""
        if not self.confidence_ranges:
            self.confidence_ranges = {
                (0, 2): 0.45,    # Confidence 0-2 → 45% expected accuracy
                (3, 4): 0.55,    # Confidence 3-4 → 55% expected accuracy
                (5, 6): 0.65,    # Confidence 5-6 → 65% expected accuracy
                (7, 8): 0.75,    # Confidence 7-8 → 75% expected accuracy
                (9, 10): 0.85,   # Confidence 9-10 → 85% expected accuracy
            }
    
    def get_expected_accuracy(self, confidence: float) -> float:
        """Get expected accuracy for a confidence score"""
        confidence_int = int(confidence)
        for (low, high), accuracy in self.confidence_ranges.items():
            if low <= confidence_int <= high:
                return accuracy
        return 0.5  # Default fallback
    
    def adjust_confidence(self, base_confidence: float, historical_accuracy: float) -> float:
        """
        Adjust confidence based on historical performance.
        
        Formula:
          - Get expected accuracy for base confidence
          - Calculate adjustment factor = historical_accuracy / expected_accuracy
          - Apply: adjusted = base_confidence * adjustment_factor
          - Clamp to 0-10 range
        
        Args:
            base_confidence: Original confidence score (0-10)
            historical_accuracy: Actual accuracy from historical data (0-1.0)
        
        Returns:
            Adjusted confidence score (0-10)
        """
        if historical_accuracy == 0:
            return 0.0
        
        expected_accuracy = self.get_expected_accuracy(base_confidence)
        if expected_accuracy == 0:
            return base_confidence
        
        # Calculate adjustment factor
        adjustment_factor = historical_accuracy / expected_accuracy
        
        # Apply adjustment
        adjusted = base_confidence * adjustment_factor
        
        # Clamp between 0-10
        return min(max(adjusted, 0.0), 10.0)


# ============================================================================
# MAIN DATABASE CLASS
# ============================================================================

class PatternAccuracyDatabase:
    """
    In-memory database for historical pattern accuracy tracking.
    
    Stores statistics for each (pattern_name, market_regime) combination.
    Used by signal_validator.py to query accuracy and calibrate confidence.
    """
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize the pattern accuracy database.
        
        Args:
            logger_instance: Optional logger instance (uses root logger if not provided)
        """
        # Storage: patterns[pattern_name][market_regime] = PatternStats
        self.patterns: Dict[str, Dict[MarketRegime, PatternStats]] = {}
        
        # Confidence calibration system
        self.calibration = ConfidenceCalibration()
        
        # Logging
        self.logger = logger_instance or logging.getLogger(__name__)
        self.logger.info("PatternAccuracyDatabase initialized")
    
    def add_pattern_result(
        self,
        pattern_name: str,
        regime: MarketRegime,
        won: bool,
        rrr: float,
        pnl: float,
    ) -> None:
        """
        Record a pattern result from a completed trade.
        
        Args:
            pattern_name: Name of the pattern (e.g., "bullish_engulfing")
            regime: Market regime at time of trade
            won: Whether trade was profitable
            rrr: Achieved risk-reward ratio
            pnl: P&L as percentage
        """
        try:
            # Ensure pattern exists
            if pattern_name not in self.patterns:
                self.patterns[pattern_name] = {}
            
            # Ensure regime exists for this pattern
            if regime not in self.patterns[pattern_name]:
                self.patterns[pattern_name][regime] = PatternStats(
                    pattern_name=pattern_name,
                    regime=regime,
                )
            
            # Update statistics
            stats = self.patterns[pattern_name][regime]
            stats.total_occurrences += 1
            
            if won:
                stats.winning_occurrences += 1
            else:
                stats.losing_occurrences += 1
            
            stats.rrr_values.append(rrr)
            stats.pnl_values.append(pnl)
            stats.last_updated = datetime.now()
            
            self.logger.debug(
                f"Recorded {pattern_name} in {regime.value}: "
                f"Win={won}, RRR={rrr:.2f}, P&L={pnl:.2f}%"
            )
        
        except Exception as e:
            self.logger.error(f"Error adding pattern result: {e}")
    
    def get_pattern_accuracy(
        self,
        pattern_name: str,
        regime: Optional[MarketRegime] = None,
        min_samples: int = 10,
        min_accuracy: float = 0.70,
    ) -> Optional[Dict[str, Any]]:
        """
        Get accuracy statistics for a pattern (optionally filtered by regime).
        
        Returns:
            Dict with:
            - accuracy: Win rate
            - samples: Sample count
            - statistically_significant: Bool
            - best_rrr: Best achieved ratio
            - worst_rrr: Worst achieved ratio
            - avg_rrr: Average ratio
            
            Or None if pattern/regime not found
        """
        try:
            # Pattern not in database
            if pattern_name not in self.patterns:
                return None
            
            # If regime specified, return that regime only
            if regime:
                if regime not in self.patterns[pattern_name]:
                    return None
                
                stats = self.patterns[pattern_name][regime]
                return {
                    'accuracy': stats.win_rate(),
                    'samples': stats.sample_size(),
                    'statistically_significant': stats.is_statistically_significant(min_samples),
                    'best_rrr': stats.best_rrr(),
                    'worst_rrr': stats.worst_rrr(),
                    'avg_rrr': stats.avg_rrr(),
                }
            
            # If no regime specified, aggregate across all regimes
            overall_stats = self.get_pattern_overall_accuracy(pattern_name)
            if overall_stats:
                return {
                    'accuracy': overall_stats.win_rate(),
                    'samples': overall_stats.sample_size(),
                    'statistically_significant': overall_stats.is_statistically_significant(min_samples),
                    'best_rrr': overall_stats.best_rrr(),
                    'worst_rrr': overall_stats.worst_rrr(),
                    'avg_rrr': overall_stats.avg_rrr(),
                }
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error getting pattern accuracy: {e}")
            return None
    
    def get_pattern_overall_accuracy(self, pattern_name: str) -> Optional[PatternStats]:
        """
        Get aggregated accuracy for a pattern across ALL market regimes.
        
        Returns:
            PatternStats with combined statistics
            Or None if pattern not found
        """
        try:
            if pattern_name not in self.patterns:
                return None
            
            # Aggregate stats across all regimes
            overall_stats = PatternStats(
                pattern_name=pattern_name,
                regime=MarketRegime.RANGE,  # Use RANGE as placeholder
            )
            
            for regime, stats in self.patterns[pattern_name].items():
                overall_stats.total_occurrences += stats.total_occurrences
                overall_stats.winning_occurrences += stats.winning_occurrences
                overall_stats.losing_occurrences += stats.losing_occurrences
                overall_stats.rrr_values.extend(stats.rrr_values)
                overall_stats.pnl_values.extend(stats.pnl_values)
            
            return overall_stats if overall_stats.total_occurrences > 0 else None
        
        except Exception as e:
            self.logger.error(f"Error getting overall pattern accuracy: {e}")
            return None
    
    def should_send_alert(
        self,
        pattern_name: str,
        regime: MarketRegime,
        confidence: float,
        min_accuracy: float = 0.70,
        min_samples: int = 10,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if signal should be sent based on historical performance.
        
        Returns:
            Tuple[bool, Dict]:
            - bool: Whether to send alert
            - Dict: Reason, accuracy, samples, adjusted_confidence
        """
        try:
            stats = self.get_pattern_accuracy(pattern_name, regime, min_samples)
            
            # No historical data yet (training mode)
            if stats is None:
                return True, {
                    'reason': 'No historical data available (training mode)',
                    'accuracy': None,
                    'samples': 0,
                    'adjusted_confidence': confidence,
                    'statistically_significant': False,
                }
            
            # Check sample size
            if not stats.get('statistically_significant', False):
                return True, {
                    'reason': f"Insufficient samples ({stats['samples']} < {min_samples})",
                    'accuracy': stats['accuracy'],
                    'samples': stats['samples'],
                    'adjusted_confidence': confidence,
                    'statistically_significant': False,
                }
            
            # Check accuracy threshold
            if stats['accuracy'] < min_accuracy:
                adjusted_conf = self.calibration.adjust_confidence(confidence, stats['accuracy'])
                return False, {
                    'reason': f"Accuracy below threshold ({stats['accuracy']:.1%} < {min_accuracy:.1%})",
                    'accuracy': stats['accuracy'],
                    'samples': stats['samples'],
                    'adjusted_confidence': adjusted_conf,
                    'statistically_significant': stats['statistically_significant'],
                }
            
            # All checks passed
            adjusted_conf = self.calibration.adjust_confidence(confidence, stats['accuracy'])
            return True, {
                'reason': f"Historical validation passed ({stats['accuracy']:.1%} accuracy, {stats['samples']} samples)",
                'accuracy': stats['accuracy'],
                'samples': stats['samples'],
                'adjusted_confidence': adjusted_conf,
                'statistically_significant': stats['statistically_significant'],
                'best_rrr': stats.get('best_rrr'),
                'worst_rrr': stats.get('worst_rrr'),
                'avg_rrr': stats.get('avg_rrr'),
            }
        
        except Exception as e:
            self.logger.error(f"Error in should_send_alert: {e}")
            return True, {
                'reason': f"Error checking historical data: {str(e)}",
                'accuracy': None,
                'samples': 0,
                'adjusted_confidence': confidence,
                'statistically_significant': False,
            }
    
    def get_all_patterns(self) -> List[str]:
        """Get list of all tracked pattern names"""
        return list(self.patterns.keys())
    
    def get_regimes_for_pattern(self, pattern_name: str) -> List[MarketRegime]:
        """Get all market regimes for which a pattern has data"""
        if pattern_name not in self.patterns:
            return []
        return list(self.patterns[pattern_name].keys())
    
    def export_to_json(self, filepath: str) -> bool:
        """
        Export all pattern data to JSON file.
        
        Returns:
            bool: Success status
        """
        try:
            export_data = {}
            
            for pattern_name, regimes_data in self.patterns.items():
                export_data[pattern_name] = {}
                for regime, stats in regimes_data.items():
                    export_data[pattern_name][regime.value] = stats.to_dict()
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Pattern accuracy data exported to {filepath}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to export pattern data: {e}")
            return False
    
    def load_from_json(self, filepath: str) -> bool:
        """
        Load pattern data from JSON file.
        
        Returns:
            bool: Success status
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            for pattern_name, regimes_data in data.items():
                if pattern_name not in self.patterns:
                    self.patterns[pattern_name] = {}
                
                for regime_str, stats_dict in regimes_data.items():
                    try:
                        regime = MarketRegime[regime_str]
                        
                        stats = PatternStats(
                            pattern_name=pattern_name,
                            regime=regime,
                            total_occurrences=stats_dict.get('total_occurrences', 0),
                            winning_occurrences=stats_dict.get('winning_occurrences', 0),
                            losing_occurrences=stats_dict.get('losing_occurrences', 0),
                            rrr_values=stats_dict.get('rrr_values', []),
                            pnl_values=stats_dict.get('pnl_values', []),
                        )
                        
                        self.patterns[pattern_name][regime] = stats
                    
                    except (KeyError, ValueError) as e:
                        self.logger.warning(f"Skipping invalid regime {regime_str}: {e}")
                        continue
            
            self.logger.info(f"Pattern accuracy data loaded from {filepath}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to load pattern data: {e}")
            return False
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for all patterns in database"""
        try:
            summary = {
                'total_patterns': len(self.patterns),
                'total_records': 0,
                'patterns': {},
                'generated_at': datetime.now().isoformat(),
            }
            
            for pattern_name, regimes_data in self.patterns.items():
                pattern_summary = {
                    'regimes': len(regimes_data),
                    'total_occurrences': 0,
                    'overall_win_rate': 0.0,
                    'expected_value': 0.0,
                    'profit_factor': 0.0,
                }
                
                # Get aggregated stats
                all_regimes_stats = self.get_pattern_overall_accuracy(pattern_name)
                
                if all_regimes_stats:
                    pattern_summary['total_occurrences'] = all_regimes_stats.total_occurrences
                    pattern_summary['overall_win_rate'] = round(all_regimes_stats.win_rate(), 4)
                    pattern_summary['expected_value'] = round(all_regimes_stats.expected_value(), 3)
                    pattern_summary['profit_factor'] = round(all_regimes_stats.profit_factor(), 3)
                    summary['total_records'] += all_regimes_stats.total_occurrences
                
                summary['patterns'][pattern_name] = pattern_summary
            
            return summary
        
        except Exception as e:
            self.logger.error(f"Error getting summary statistics: {e}")
            return {'error': str(e)}
    
    def get_pattern_report(self, pattern_name: str) -> Dict[str, Any]:
        """
        Get detailed report for a specific pattern across all regimes.
        
        Returns:
            Dict with pattern_name, per-regime stats, and overall stats
        """
        try:
            if pattern_name not in self.patterns:
                return {'error': f'Pattern {pattern_name} not found'}
            
            report = {
                'pattern_name': pattern_name,
                'regimes': {},
                'generated_at': datetime.now().isoformat(),
            }
            
            # Per-regime stats
            for regime, stats in self.patterns[pattern_name].items():
                report['regimes'][regime.value] = stats.to_dict()
            
            # Overall stats
            overall = self.get_pattern_overall_accuracy(pattern_name)
            if overall:
                report['overall'] = overall.to_dict()
            
            return report
        
        except Exception as e:
            self.logger.error(f"Error getting pattern report: {e}")
            return {'error': str(e)}
    
    def clear_pattern_data(self, pattern_name: str) -> bool:
        """
        Clear all data for a specific pattern.
        
        Returns:
            bool: Success status
        """
        try:
            if pattern_name in self.patterns:
                del self.patterns[pattern_name]
                self.logger.info(f"Cleared data for pattern: {pattern_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error clearing pattern data: {e}")
            return False
    
    def clear_all(self) -> bool:
        """
        Clear all data from database.
        
        Returns:
            bool: Success status
        """
        try:
            self.patterns = {}
            self.logger.info("Cleared all pattern accuracy data")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing database: {e}")
            return False


# ============================================================================
# SINGLETON INSTANCE (OPTIONAL)
# ============================================================================

_accuracy_db_instance = None


def get_accuracy_db() -> PatternAccuracyDatabase:
    """
    Get or create singleton instance of the accuracy database.
    
    Useful for applications that want a global database instance.
    
    Returns:
        PatternAccuracyDatabase: Singleton instance
    """
    global _accuracy_db_instance
    
    if _accuracy_db_instance is None:
        _accuracy_db_instance = PatternAccuracyDatabase()
    
    return _accuracy_db_instance


# ============================================================================
# MAIN: TEST DATABASE
# ============================================================================

if __name__ == "__main__":
    # Test the database
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create database
        db = PatternAccuracyDatabase()
        print("✓ PatternAccuracyDatabase initialized")
        
        # Add some test data
        db.add_pattern_result(
            "bullish_engulfing",
            MarketRegime.UPTREND,
            won=True,
            rrr=2.1,
            pnl=5.2
        )
        print("✓ Test data added")
        
        # Query data
        result = db.get_pattern_accuracy("bullish_engulfing", MarketRegime.UPTREND)
        print(f"✓ Query successful: {result}")
        
        # Get summary
        summary = db.get_summary_statistics()
        print(f"✓ Summary: {summary}")
        
        print("✓ PatternAccuracyDatabase ready for production")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
