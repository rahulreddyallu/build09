# signals_db.py
# Historical Pattern Accuracy Database
# Stores and manages pattern accuracy data by market regime
# Used for pre-alert validation and confidence calibration

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification"""
    STRONG_UPTREND = "STRONG_UPTREND"
    UPTREND = "UPTREND"
    WEAK_UPTREND = "WEAK_UPTREND"
    RANGE = "RANGE"
    WEAK_DOWNTREND = "WEAK_DOWNTREND"
    DOWNTREND = "DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"


@dataclass
class PatternStats:
    """Statistics for a pattern in a specific regime"""
    pattern_name: str
    regime: MarketRegime
    
    # Performance metrics
    total_occurrences: int = 0
    winning_occurrences: int = 0
    losing_occurrences: int = 0
    
    # RRR tracking
    rrr_values: List[float] = field(default_factory=list)
    
    # P&L tracking
    pnl_values: List[float] = field(default_factory=list)
    
    # Meta
    last_updated: datetime = field(default_factory=datetime.now)
    
    def win_rate(self) -> float:
        """Calculate win rate percentage"""
        if self.total_occurrences == 0:
            return 0.0
        return self.winning_occurrences / self.total_occurrences
    
    def best_rrr(self) -> float:
        """Best RRR achieved"""
        return max(self.rrr_values) if self.rrr_values else 0.0
    
    def worst_rrr(self) -> float:
        """Worst RRR achieved"""
        return min(self.rrr_values) if self.rrr_values else 0.0
    
    def avg_rrr(self) -> float:
        """Average RRR"""
        return np.mean(self.rrr_values) if self.rrr_values else 0.0
    
    def avg_pnl(self) -> float:
        """Average P&L percentage"""
        return np.mean(self.pnl_values) if self.pnl_values else 0.0
    
    def std_dev_pnl(self) -> float:
        """Standard deviation of P&L"""
        return float(np.std(self.pnl_values)) if len(self.pnl_values) > 1 else 0.0
    
    def sample_size(self) -> int:
        """Get sample size"""
        return self.total_occurrences
    
    def is_statistically_significant(self, min_samples: int = 10) -> bool:
        """Check if sample size is statistically significant"""
        return self.sample_size() >= min_samples
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export"""
        return {
            'pattern_name': self.pattern_name,
            'regime': self.regime.value,
            'total_occurrences': self.total_occurrences,
            'winning_occurrences': self.winning_occurrences,
            'losing_occurrences': self.losing_occurrences,
            'win_rate': self.win_rate(),
            'best_rrr': self.best_rrr(),
            'worst_rrr': self.worst_rrr(),
            'avg_rrr': self.avg_rrr(),
            'avg_pnl': self.avg_pnl(),
            'std_dev_pnl': self.std_dev_pnl(),
            'sample_size': self.sample_size(),
            'statistically_significant': self.is_statistically_significant(),
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class ConfidenceCalibration:
    """Calibration mapping confidence scores to historical accuracy"""
    confidence_ranges: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with default calibration"""
        if not self.confidence_ranges:
            self.confidence_ranges = {
                (0, 2): 0.45,      # 1-2 confidence → 45% expected accuracy
                (3, 4): 0.55,      # 3-4 confidence → 55% expected accuracy
                (5, 6): 0.65,      # 5-6 confidence → 65% expected accuracy
                (7, 8): 0.75,      # 7-8 confidence → 75% expected accuracy
                (9, 10): 0.85,     # 9-10 confidence → 85% expected accuracy
            }
    
    def get_expected_accuracy(self, confidence: float) -> float:
        """Get expected accuracy for a confidence score"""
        confidence_int = int(confidence)
        for (low, high), accuracy in self.confidence_ranges.items():
            if low <= confidence_int <= high:
                return accuracy
        return 0.5  # Default fallback
    
    def adjust_confidence(self, base_confidence: float, 
                         historical_accuracy: float) -> float:
        """Adjust confidence based on historical performance"""
        if historical_accuracy == 0:
            return 0.0
        
        expected_accuracy = self.get_expected_accuracy(base_confidence)
        if expected_accuracy == 0:
            return base_confidence
        
        # Calculate adjustment factor
        adjustment_factor = historical_accuracy / expected_accuracy
        
        # Apply adjustment with bounds
        adjusted = base_confidence * adjustment_factor
        return min(max(adjusted, 0.0), 10.0)  # Clamp between 0-10


class PatternAccuracyDatabase:
    """In-memory database for pattern accuracy tracking"""
    
    def __init__(self):
        """Initialize the pattern accuracy database"""
        self.patterns: Dict[str, Dict[MarketRegime, PatternStats]] = {}
        self.calibration = ConfidenceCalibration()
        self.logger = logging.getLogger(__name__)
        self.logger.info("PatternAccuracyDatabase initialized")
    
    def add_pattern_result(self, pattern_name: str, regime: MarketRegime,
                          won: bool, rrr: float, pnl: float) -> None:
        """Record a pattern result"""
        key = pattern_name
        
        if key not in self.patterns:
            self.patterns[key] = {}
        
        if regime not in self.patterns[key]:
            self.patterns[key][regime] = PatternStats(
                pattern_name=pattern_name,
                regime=regime
            )
        
        stats = self.patterns[key][regime]
        stats.total_occurrences += 1
        
        if won:
            stats.winning_occurrences += 1
        else:
            stats.losing_occurrences += 1
        
        stats.rrr_values.append(rrr)
        stats.pnl_values.append(pnl)
        stats.last_updated = datetime.now()
    
    def get_pattern_accuracy(self, pattern_name: str, 
                            regime: MarketRegime) -> Optional[PatternStats]:
        """Get accuracy stats for a specific pattern in a regime"""
        if pattern_name not in self.patterns:
            return None
        
        if regime not in self.patterns[pattern_name]:
            return None
        
        return self.patterns[pattern_name][regime]
    
    def get_pattern_overall_accuracy(self, pattern_name: str) -> Optional[PatternStats]:
        """Get overall accuracy for a pattern across all regimes"""
        if pattern_name not in self.patterns:
            return None
        
        # Aggregate stats across all regimes
        overall_stats = PatternStats(
            pattern_name=pattern_name,
            regime=MarketRegime.RANGE  # Use RANGE as placeholder
        )
        
        for regime, stats in self.patterns[pattern_name].items():
            overall_stats.total_occurrences += stats.total_occurrences
            overall_stats.winning_occurrences += stats.winning_occurrences
            overall_stats.losing_occurrences += stats.losing_occurrences
            overall_stats.rrr_values.extend(stats.rrr_values)
            overall_stats.pnl_values.extend(stats.pnl_values)
        
        return overall_stats if overall_stats.total_occurrences > 0 else None
    
    def should_send_alert(self, pattern_name: str, regime: MarketRegime,
                         confidence: float, min_accuracy: float = 0.70,
                         min_samples: int = 10) -> Tuple[bool, Dict]:
        """Determine if alert should be sent based on historical performance"""
        stats = self.get_pattern_accuracy(pattern_name, regime)
        
        # Default: send if no historical data yet (training mode)
        if stats is None:
            return True, {
                'reason': 'No historical data available (training mode)',
                'accuracy': None,
                'samples': 0,
                'adjusted_confidence': confidence
            }
        
        # Check sample size
        if not stats.is_statistically_significant(min_samples):
            return True, {
                'reason': f'Insufficient samples ({stats.sample_size()} < {min_samples})',
                'accuracy': stats.win_rate(),
                'samples': stats.sample_size(),
                'adjusted_confidence': confidence
            }
        
        # Check accuracy threshold
        if stats.win_rate() < min_accuracy:
            return False, {
                'reason': f'Accuracy below threshold ({stats.win_rate():.1%} < {min_accuracy:.1%})',
                'accuracy': stats.win_rate(),
                'samples': stats.sample_size(),
                'adjusted_confidence': self.calibration.adjust_confidence(
                    confidence, stats.win_rate()
                )
            }
        
        # Adjust confidence based on historical performance
        adjusted_confidence = self.calibration.adjust_confidence(
            confidence, stats.win_rate()
        )
        
        return True, {
            'reason': 'Historical validation passed',
            'accuracy': stats.win_rate(),
            'samples': stats.sample_size(),
            'adjusted_confidence': adjusted_confidence,
            'best_rrr': stats.best_rrr(),
            'worst_rrr': stats.worst_rrr(),
            'avg_rrr': stats.avg_rrr()
        }
    
    def get_all_patterns(self) -> List[str]:
        """Get list of all tracked patterns"""
        return list(self.patterns.keys())
    
    def get_regimes_for_pattern(self, pattern_name: str) -> List[MarketRegime]:
        """Get all regimes for which a pattern has data"""
        if pattern_name not in self.patterns:
            return []
        return list(self.patterns[pattern_name].keys())
    
    def export_to_json(self, filepath: str) -> None:
        """Export all pattern data to JSON"""
        export_data = {}
        
        for pattern_name, regimes_data in self.patterns.items():
            export_data[pattern_name] = {}
            for regime, stats in regimes_data.items():
                export_data[pattern_name][regime.value] = stats.to_dict()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            self.logger.info(f"Pattern accuracy data exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export pattern data: {e}")
    
    def load_from_json(self, filepath: str) -> None:
        """Load pattern data from JSON"""
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
                            pnl_values=stats_dict.get('pnl_values', [])
                        )
                        self.patterns[pattern_name][regime] = stats
                    except (KeyError, ValueError):
                        continue
            
            self.logger.info(f"Pattern accuracy data loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load pattern data: {e}")
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for all patterns"""
        summary = {
            'total_patterns': len(self.patterns),
            'total_records': 0,
            'patterns': {}
        }
        
        for pattern_name, regimes_data in self.patterns.items():
            pattern_summary = {
                'regimes': len(regimes_data),
                'total_occurrences': 0,
                'overall_win_rate': 0.0
            }
            
            all_regimes_stats = self.get_pattern_overall_accuracy(pattern_name)
            if all_regimes_stats:
                pattern_summary['total_occurrences'] = all_regimes_stats.total_occurrences
                pattern_summary['overall_win_rate'] = all_regimes_stats.win_rate()
                summary['total_records'] += all_regimes_stats.total_occurrences
            
            summary['patterns'][pattern_name] = pattern_summary
        
        return summary
    
    def get_pattern_report(self, pattern_name: str) -> Dict:
        """Get detailed report for a specific pattern"""
        if pattern_name not in self.patterns:
            return {'error': f'Pattern {pattern_name} not found'}
        
        report = {
            'pattern_name': pattern_name,
            'regimes': {}
        }
        
        for regime, stats in self.patterns[pattern_name].items():
            report['regimes'][regime.value] = stats.to_dict()
        
        # Add overall stats
        overall = self.get_pattern_overall_accuracy(pattern_name)
        if overall:
            report['overall'] = overall.to_dict()
        
        return report


# Singleton instance (optional, for convenience)
_accuracy_db_instance = None


def get_accuracy_db() -> PatternAccuracyDatabase:
    """Get singleton instance of accuracy database"""
    global _accuracy_db_instance
    if _accuracy_db_instance is None:
        _accuracy_db_instance = PatternAccuracyDatabase()
    return _accuracy_db_instance
