from typing import ClassVar, Optional
from dataclasses import dataclass, asdict, fields, field
from contextlib import contextmanager
import numpy as np

import pandas as pd


@dataclass
class DataFields3D:
    x: str
    y: str
    z: str
    weights: Optional[str] = None
    spatial: ClassVar[tuple[str]] = ('x', 'y')
    temporal: ClassVar[tuple[str]] = ('z',)

    @property
    def names(self):
        return [field.name for field in fields(self)]

    @property
    def labels(self):
        return [getattr(self, field.name) for field in fields(self)]
    
    @property
    def fields(self):
        return asdict(self)


class Dataset:
    def __init__(
            self,
            data: pd.DataFrame,
            fields: DataFields3D,
            data_format: Optional[str] = None,
            is_persistent: bool = True,
            name: Optional[str] = None
        ):
        self.fields = fields
        self._data_container = {'source': data}
        self.default_format = data_format
        self._format_kwargs = {}
        self.is_persistent = is_persistent # cache calculated data in specified formats
        self.initialize_formats([self.default_format])
        if name is None:
            name = self.__class__.__name__
        self.name = name
    
    def initialize_formats(self, formats):
        if self.is_persistent:
            for format in formats:
                assert self.get_formatted_data(format) is not None

    @property
    def data(self):
        return self.get_formatted_data(self.default_format)

    def get_formatted_data(self, format):
        if format is None: # return original data if format is not specified
            return self._data_container['source']
        format_kwargs = {}
        if isinstance(format, (tuple, list)):
            format, format_kwargs = format
            if self._format_kwargs.get(format, None) != format_kwargs:
                # cleanup cached data - it was obtained with different settings
                self.cleanup(format)
            self._format_kwargs[format] = format_kwargs
        try:
            formatted_data = self._data_container[format]
        except KeyError: # no data in this format -> generate it from defaults
            data_formatter = DataFormatter(format)
            formatted_data = data_formatter(self, **format_kwargs)
            if self.is_persistent: # store data in requested format permanently
                self._data_container[format] = formatted_data
        return formatted_data
    
    @contextmanager
    def format(self, format: str):
        # store current values of data formats
        default_format = self.default_format
         # temporarily set new data formats
        self.default_format = format
        try:
            yield self
        finally: # restore initial values
            self.default_format = default_format
    
    def has_format(self, format: str):
        try:
            data = self._data_container[format]
        except KeyError:
            return False
        return data is not None

    def cleanup(self, formats=None):
        if formats is None:
            formats = list(self._data_container.keys())
        if not isinstance(formats, (list, set, tuple)):
            formats = [formats]
        for format in formats:
            if format != 'source': # leave source data intact
                if format in self._data_container:
                    del self._data_container[format]
                if format in self._format_kwargs:
                    del self._format_kwargs[format]


class DataFormatter:
    def __init__(self, default_format: Optional[str]=None):
        self.default_format = default_format or 'spatio_temporal_tensor'
        self._formatters = {
            'spatio_temporal_tensor': dataframe_to_spatiotemporal,
        }
    
    def __call__(self, dataset: Dataset, format: Optional[str]=None, **kwargs):
        return self.format(dataset, format, **kwargs) 

    def format(self, dataset: Dataset, format: str, **kwargs):
        if format is None:
            format = self.default_format
        try:
            formatter = self._formatters[format]
        except KeyError:
            raise NotImplementedError(f'Unrecognized format: {format}')
        data = dataset._data_container['source']
        # TODO allow to format data based on alternative formats other than source
        return formatter(data, dataset.fields, **kwargs)
    
    def register(self, format, formatter):
        self._formatters[format] = formatter


def dataframe_to_spatiotemporal(data: pd.DataFrame, data_fields: DataFields3D, inplace=False):
    data_index = {}
    data_codes = {}
    idx_fields = data_fields.spatial + data_fields.temporal
    idx_cols = []
    for field_name in idx_fields:
        label = getattr(data_fields, field_name)
        idx_cols.append(label)
        codes, index = pd.factorize(data[label], sort=True)
        data_index[field_name] = pd.Index(index, name=label)
        if inplace:
            data.loc[:, label] = codes
            continue
        data_codes[label] = codes
    if data_codes:
        data = data.assign(**data_codes)
    # tensor data
    shape = tuple(len(data_index[name]) for name in idx_fields)
    idx = data[idx_cols].values
    if data_fields.weights is None:
        vals = np.ones(idx.shape[0])
    else:
        vals = data[data_fields.weights].values
    return idx, vals, shape, data_index
