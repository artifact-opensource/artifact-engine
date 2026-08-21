#!/usr/bin/env python3
"""
🌊 Data Streamer - Smart data ingestion and normalization
Streams data from APIs, online sources, and virtually anything
"""

import asyncio
import json
import aiohttp
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import re
from datetime import datetime

@dataclass
class DataSource:
    """Data source configuration"""
    url: str
    type: str  # api, csv, json, text, sql, database, file
    format: str  # json, csv, txt, parquet, etc.
    auth_token: Optional[str] = None
    headers: Dict[str, str] = None
    params: Dict[str, Any] = None

@dataclass
class DataStream:
    """Streamed data with metadata"""
    data: Any
    source: str
    detected_type: str
    normalized: bool
    timestamp: datetime
    size: int
    quality_score: float

class DataStreamer:
    """Smart data streaming and normalization engine"""
    
    def __init__(self):
        self.session = None
        self.supported_formats = ['json', 'csv', 'txt', 'parquet', 'xml', 'html']
        self.detection_patterns = {
            'json': [r'^\s*[\{\[]', r'"key":', r':\s*"value"'],
            'csv': [r',.*\.+', r'^[^,]+,[^,]+,[^,]+$'],
            'xml': [r'<\w+>.*</\w+>', r'<\?xml'],
            'html': [r'<html>', r'<body>', r'<div>'],
            'text': [r'^[A-Za-z]', r'\w+\s+\w+']
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def stream_from_api(self, url: str, **kwargs) -> DataStream:
        """Stream data from API endpoint"""
        headers = kwargs.get('headers', {})
        params = kwargs.get('params', {})
        
        async with self.session.get(url, headers=headers, params=params) as response:
            data = await response.json()
            
        return DataStream(
            data=data,
            source=url,
            detected_type='json',
            normalized=True,
            timestamp=datetime.now(),
            size=len(str(data)),
            quality_score=self._calculate_quality(data)
        )
    
    async def stream_from_url(self, url: str) -> DataStream:
        """Stream data from any URL"""
        async with self.session.get(url) as response:
            content_type = response.headers.get('content-type', '')
            text = await response.text()
            
        detected_type = self._detect_type(text)
        data = self._parse_content(text, detected_type)
        
        return DataStream(
            data=data,
            source=url,
            detected_type=detected_type,
            normalized=True,
            timestamp=datetime.now(),
            size=len(text),
            quality_score=self._calculate_quality(data)
        )
    
    async def stream_from_file(self, filepath: str) -> DataStream:
        """Stream data from local file"""
        path = Path(filepath)
        ext = path.suffix.lower()
        
        if ext == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
            detected_type = 'json'
        elif ext == '.csv':
            df = pd.read_csv(path)
            data = df.to_dict('records')
            detected_type = 'csv'
        elif ext == '.parquet':
            df = pd.read_parquet(path)
            data = df.to_dict('records')
            detected_type = 'parquet'
        else:
            with open(path, 'r') as f:
                data = f.read()
            detected_type = 'text'
        
        return DataStream(
            data=data,
            source=filepath,
            detected_type=detected_type,
            normalized=True,
            timestamp=datetime.now(),
            size=len(str(data)),
            quality_score=self._calculate_quality(data)
        )
    
    async def stream_from_database(self, connection_string: str, query: str) -> DataStream:
        """Stream data from database"""
        # Would need actual DB connection
        # This is a placeholder
        return DataStream(
            data=[],
            source=connection_string,
            detected_type='database',
            normalized=True,
            timestamp=datetime.now(),
            size=0,
            quality_score=0.0
        )
    
    def _detect_type(self, text: str) -> str:
        """Intelligent data type detection"""
        for data_type, patterns in self.detection_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return data_type
        return 'text'
    
    def _parse_content(self, text: str, data_type: str) -> Any:
        """Parse content based on detected type"""
        try:
            if data_type == 'json':
                return json.loads(text)
            elif data_type == 'csv':
                from io import StringIO
                df = pd.read_csv(StringIO(text))
                return df.to_dict('records')
            else:
                return text
        except Exception as e:
            return {'raw_text': text, 'parse_error': str(e)}
    
    def _calculate_quality(self, data: Any) -> float:
        """Calculate data quality score"""
        if isinstance(data, dict):
            return min(1.0, len(data) / 100)
        elif isinstance(data, list):
            return min(1.0, len(data) / 1000)
        elif isinstance(data, str):
            return min(1.0, len(data) / 10000)
        else:
            return 0.5
    
    async def normalize(self, data_stream: DataStream) -> DataStream:
        """Normalize data to standard format"""
        if data_stream.normalized:
            return data_stream
        
        normalized_data = self._normalize_data(data_stream.data, data_stream.detected_type)
        
        return DataStream(
            data=normalized_data,
            source=data_stream.source,
            detected_type=data_stream.detected_type,
            normalized=True,
            timestamp=data_stream.timestamp,
            size=data_stream.size,
            quality_score=data_stream.quality_score
        )
    
    def _normalize_data(self, data: Any, data_type: str) -> List[Dict]:
        """Normalize data to list of dictionaries"""
        if isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                return data
            else:
                return [{'text': str(item)} for item in data]
        elif isinstance(data, dict):
            return [data]
        else:
            return [{'text': str(data)}]

# Example usage
async def main():
    """Example data streaming"""
    async with DataStreamer() as streamer:
        # Stream from API
        api_data = await streamer.stream_from_api(
            "https://api.example.com/data",
            headers={"Authorization": "Bearer token"}
        )
        
        print(f"API Data: {api_data.data}")
        print(f"Quality Score: {api_data.quality_score}")

if __name__ == "__main__":
    asyncio.run(main())