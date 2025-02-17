# Copyright (c) MarsLight Studio
# This file is similar to file_storage to retrieve data from http

from pathlib import Path
from typing import Iterable, Union, Dict, Mapping, Tuple, List
import os
import platform
import requests
import json
from functools import lru_cache
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading

import numpy as np
import pandas as pd

from qlib.utils.time import Freq
from qlib.utils.resam import resam_calendar
from qlib.config import C
from qlib.data.cache import H
from qlib.data.data import Cal
from qlib.log import get_module_logger
from qlib.data.storage import CalendarStorage, InstrumentStorage, FeatureStorage, CalVT, InstKT, InstVT

logger = get_module_logger("http_storage")

class HttpStorageMixin:
    """
    Mixin class for http storage.
    """
    @property
    def http_uri(self):
        return C["http_uri"] if getattr(self, "_http_uri", None) is None else self._http_uri
    
    @property
    def supported_freq(self):
        _v = "_supported_freq"
        if hasattr(self, _v):
            return getattr(self, _v)
        periods = ["1min", "5min", "15min", "30min", "day"]
        freq_l = [Freq(p) for p in periods]
        setattr(self, _v, freq_l)
        return freq_l
    
    @property
    def cache_path(self):
        """Get the cache path based: ~/.cache/qlib_cache/<storage_name>"""
        base_path = os.path.join(os.path.expanduser("~"), ".cache", "qlib_cache")
        
        os.makedirs(base_path, exist_ok=True)
        return base_path

    def _get_cache_file(self, *args):
        """Generate cache file path based on arguments"""
        cache_dir = os.path.join(self.cache_path, self.storage_name)
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "_".join(map(str, args)) + ".cache")

class HttpFeatureStorage(HttpStorageMixin, FeatureStorage):
    """
    HTTP Feature Storage
    """
    def __init__(self, instrument: str, field: str, freq: str, **kwargs):
        super().__init__(instrument=instrument, field=field, freq=freq, **kwargs)
        self._data = None
        # Map qlib frequency to API period format
        self.freq_map = {
            "1min": "1m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "day": "1d"
        }

    def feature(self, instrument: str, field: str, start_time: pd.Timestamp, end_time: pd.Timestamp, freq: str):
        """Get raw feature data through HTTP or cache"""
        period = self.freq_map.get(freq, "1d")
        cache_file = self._get_cache_file(instrument,
                                        start_time.strftime("%Y%m%d"), 
                                        end_time.strftime("%Y%m%d"), 
                                        freq)
        if os.path.exists(cache_file):
            return pd.read_parquet(cache_file)

        params = {
            "tickers": instrument,
            "start_time": start_time.strftime("%Y%m%d"),
            "end_time": end_time.strftime("%Y%m%d"),
            "period": period,
            "count": "-1",
            "dividend_type": "front"
        }
        
        url = f"{self.http_uri}/quote/kline"
        res = requests.get(url, params=params)
        rows = res.json()['data'][instrument]
        
        df = pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'amount'])
        df.index = df['time'].apply(pd.Timestamp)
        df.drop('time', axis=1, inplace=True)
        df['vwap'] = df['amount'] / df['volume'] / 100

        # 找到第一个 close 为负数的位置，删除该行及之前的所有行
        negative_low_idx = df[df['low'] < 0].index
        if len(negative_low_idx) > 0:
            first_negative = negative_low_idx[-1]
            df = df[df.index > first_negative]

        # Save raw data to cache
        df.to_parquet(cache_file)
        
        return df
        
    @property
    def data(self) -> pd.Series:
        """Get processed feature data"""
        if self._data is None:
            try:
                # Get data for a sufficiently large time range
                start_time = pd.Timestamp('2010-01-01')
                end_time = pd.Timestamp.now()
                
                # Get raw data
                df = self.feature(self.instrument, self.field, start_time, end_time, self.freq)
                
                # Process the data
                df.index = Cal.locate_index_between(df.index, self.freq)

                if len(df):
                    df = df.reindex(range(df.index[0], df.index[-1] + 1))

                # Extract the requested field
                if self.field in df.columns:
                    self._data = df[self.field]
                else:
                    raise ValueError(f"Field {self.field} not found in data")
                    
            except Exception as e:
                logger.warning(f"Failed to get data: {str(e)}")
                return pd.Series(dtype=np.float32)
        return self._data

    @property
    def start_index(self) -> Union[int, None]:
        """Get start index"""
        if len(self.data) == 0:
            return None
        return self.data.index[0]

    @property
    def end_index(self) -> Union[int, None]:
        """Get end index"""
        if len(self.data) == 0:
            return None
        return self.data.index[-1]

    def clear(self) -> None:
        """Clear the storage"""
        self._data = pd.Series(dtype=np.float32)
        # 可以选择删除缓存文件
        cache_file = self._get_cache_file(self.instrument, self.field, 
                                        "20100101", "29991231", self.freq)
        if os.path.exists(cache_file):
            os.remove(cache_file)

    def write(self, data_array: Union[List, np.ndarray, Tuple], index: int = None):
        """Write data to storage"""
        if len(data_array) == 0:
            return
        
        if self._data is None:
            self._data = pd.Series(dtype=np.float32)
        
        if index is None:
            index = self.end_index + 1 if self.end_index is not None else 0
            
        # Convert data_array to Series
        new_data = pd.Series(data_array, index=range(index, index + len(data_array)))
        
        # Update existing data
        self._data = pd.concat([self._data, new_data]).sort_index()
        self._data = self._data[~self._data.index.duplicated(keep='last')]

    def __getitem__(self, i) -> Union[Tuple[int, float], pd.Series]:
        """Get item implementation"""
        if len(self.data) == 0:
            if isinstance(i, int):
                return (None, None)
            return pd.Series(dtype=np.float32)

        if isinstance(i, int):
            try:
                val = self.data.iloc[i]
                return (i, val)
            except IndexError:
                return (None, None)
        elif isinstance(i, slice):
            start = i.start if i.start is not None else 0
            stop = i.stop if i.stop is not None else len(self.data)
            try:
                return self.data.iloc[start:stop:i.step]
            except IndexError:
                return pd.Series(dtype=np.float32)

    def __len__(self) -> int:
        """Get length of storage"""
        return len(self.data)


class HttpCalendarStorage(HttpStorageMixin, CalendarStorage):
    """
    HTTP Calendar Storage
    """
    def __init__(self, freq: str, future: bool, **kwargs):
        super().__init__(freq=freq, future=future, **kwargs)
        self._data = None

    def get_calendar(self, start_time: str, end_time: str) -> List[str]:
        """Get calendar data through HTTP"""
        cache_file = self._get_cache_file(start_time, end_time)
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        url = f"{self.http_uri}/quote/calendar"
        params = {
            "start_time": start_time,
            "end_time": end_time,
            "market": "SH"
        }
        
        res = requests.get(url, params=params)
        calendar_data = res.json()['data']
        
        # Cache the results
        with open(cache_file, 'w') as f:
            json.dump(calendar_data, f)
            
        return calendar_data

    @property
    def data(self) -> List[CalVT]:
        """Get all calendar data"""
        if self._data is None:
            try:
                # 获取一个足够大的时间范围的数据
                self._data = self.get_calendar("20100101", datetime.now().strftime("%Y%m%d"))
            except Exception as e:
                logger.warning(f"Failed to get calendar data: {str(e)}")
                raise ValueError("Calendar data not available")
        return self._data

    def clear(self) -> None:
        """Clear the storage"""
        self._data = None
        cache_file = self._get_cache_file("20100101", "29991231")
        if os.path.exists(cache_file):
            os.remove(cache_file)

    def extend(self, iterable: Iterable[CalVT]) -> None:
        """Extend calendar data"""
        self._data = self.data + list(iterable)

    def index(self, value: CalVT) -> int:
        """Get index of value"""
        return self.data.index(value)

    def insert(self, index: int, value: CalVT) -> None:
        """Insert value at index"""
        self._data = self.data[:index] + [value] + self.data[index:]

    def remove(self, value: CalVT) -> None:
        """Remove value"""
        self._data = [x for x in self.data if x != value]

    def __setitem__(self, i, value) -> None:
        """Set item implementation"""
        if isinstance(i, int):
            self._data[i] = value
        elif isinstance(i, slice):
            self._data[i.start:i.stop:i.step] = value

    def __delitem__(self, i) -> None:
        """Delete item implementation"""
        if isinstance(i, int):
            del self._data[i]
        elif isinstance(i, slice):
            del self._data[i.start:i.stop:i.step]

    def __getitem__(self, i) -> Union[CalVT, List[CalVT]]:
        """Get item implementation"""
        if isinstance(i, int):
            return self.data[i]
        elif isinstance(i, slice):
            return self.data[i.start:i.stop:i.step]

    def __len__(self) -> int:
        """Get length of storage"""
        return len(self.data)


class HttpInstrumentStorage(HttpStorageMixin, InstrumentStorage):
    """
    HTTP Instrument Storage
    """
    def __init__(self, market: str, freq: str, **kwargs):
        super().__init__(market=market, freq=freq, **kwargs)
        self._data = None
        self.market_map = {
            "csi300": "沪深300",
            "hs300": "沪深300",
            "csi100": "中证100",
            "csi500": "中证500",
            "csi1000": "中证1000"
        }
        self._lock = threading.Lock()

    def _get_instrument_start_time(self, instrument: str) -> (str, str):
        """
        获取单个股票的实际开始时间
        """
        try:
            # 使用HttpFeatureStorage来获取数据
            feature_storage = HttpFeatureStorage(instrument=instrument, field='close', freq='day')
            start_time = pd.Timestamp('2010-01-01')
            end_time = pd.Timestamp.now()
            data = feature_storage.feature(instrument, 'close', start_time, end_time, 'day')
            
            if len(data) > 0:
                return (data.index[0].strftime('%Y%m%d'), 
                        data.index[-1].strftime('%Y%m%d'))
        except Exception as e:
            logger.warning(f"Failed to get start time for {instrument}: {str(e)}")
        return (end_time.strftime("%Y%m%d"), end_time.strftime("%Y%m%d"))  # 如果获取失败，返回默认时间

    def _process_instrument_time_range(self, instrument: str) -> tuple:
        """
        处理单个股票的时间范围
        """
        start_time, end_time = self._get_instrument_start_time(instrument)
        return instrument, [(pd.Timestamp(start_time), pd.Timestamp(end_time)),]
    
    def list_instruments(self, market: str, as_list: bool = False) -> Union[List[str], set]:
        """Get instrument list through HTTP"""
        cache_file = self._get_cache_file(market)
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                instruments = json.load(f)
        else:
            mapped_market = self.market_map.get(market.lower())
            if not mapped_market:
                raise ValueError(f"Unsupported market: {market}")
            
            url = f"{self.http_uri}/quote/sector/component"
            params = {"sector": mapped_market}
            res = requests.get(url, params=params)
            instruments = res.json()['data']
            
            # Cache the results
            with open(cache_file, 'w') as f:
                json.dump(instruments, f)
        
        return instruments if as_list else set(instruments)

    @property
    def data(self) -> Dict[InstKT, InstVT]:
        """Get all instrument data"""
        if self._data is None:
            with self._lock:  # 使用线程锁确保线程安全
                if self._data is None:  # 双重检查锁定模式
                    try:
                        instruments = self.list_instruments(self.market, as_list=True)
                        
                        # 使用线程池并行处理
                        with ThreadPoolExecutor(max_workers=min(32, len(instruments))) as executor:
                            # 并行获取每个股票的时间范围
                            results = list(executor.map(self._process_instrument_time_range, instruments))
                            
                            # 将结果转换为字典
                            self._data = dict(results)
                            
                    except Exception as e:
                        logger.warning(f"Failed to get instrument data: {str(e)}")
                        raise ValueError("Instrument data not available")
        return self._data

    # @property
    # def data(self) -> Dict[InstKT, InstVT]:
    #     """Get all instrument data"""
    #     if self._data is None:
    #         try:
    #             instruments = self.list_instruments(self.market, as_list=True)
    #             # 为每个instrument创建一个空的InstVT
    #             self._data = {inst: [（pd.Timestamp("20100101"), pd.Timestamp(datetime.now().strftime("%Y%m%d"))),] for inst in instruments}
    #         except Exception as e:
    #             logger.warning(f"Failed to get instrument data: {str(e)}")
    #             raise ValueError("Instrument data not available")
    #     return self._data

    def clear(self) -> None:
        """Clear the storage"""
        self._data = None
        cache_file = self._get_cache_file(self.market)
        if os.path.exists(cache_file):
            os.remove(cache_file)

    def update(self, *args, **kwargs) -> None:
        """Update storage with new data"""
        if len(args) > 0:
            other = args[0]
            if hasattr(other, 'items'):
                for k, v in other.items():
                    self._data[k] = v
            else:
                for k, v in other:
                    self._data[k] = v
        for k, v in kwargs.items():
            self._data[k] = v

    def __setitem__(self, k: InstKT, v: InstVT) -> None:
        """Set item implementation"""
        self._data[k] = v

    def __delitem__(self, k: InstKT) -> None:
        """Delete item implementation"""
        del self._data[k]

    def __getitem__(self, k: InstKT) -> InstVT:
        """Get item implementation"""
        return self.data[k]

    def __len__(self) -> int:
        """Get length of storage"""
        return len(self.data)
