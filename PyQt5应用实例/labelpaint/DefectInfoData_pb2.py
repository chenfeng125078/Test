# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: DefectInfoData.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='DefectInfoData.proto',
  package='',
  syntax='proto3',
  serialized_pb=_b('\n\x14\x44\x65\x66\x65\x63tInfoData.proto\"\x14\n\x05Image\x12\x0b\n\x03\x62uf\x18\x01 \x01(\x0c\":\n\x0f\x46\x65\x61tureInfoData\x12\x13\n\x0b\x66\x65\x61turename\x18\x01 \x01(\t\x12\x12\n\nfeatureVal\x18\x02 \x01(\x02\"\xe8\x01\n\x10SingleDefectData\x12\x0c\n\x04n_Id\x18\x01 \x01(\x05\x12\r\n\x05meter\x18\x02 \x01(\x02\x12\x12\n\ndefectType\x18\x03 \x01(\t\x12\x1b\n\x0brealTimeImg\x18\x04 \x01(\x0b\x32\x06.Image\x12\x1b\n\x0bstandardImg\x18\x05 \x01(\x0b\x32\x06.Image\x12%\n\x0b\x66\x65\x61tureData\x18\x06 \x03(\x0b\x32\x10.FeatureInfoData\x12\x14\n\x0ch_resolution\x18\x07 \x01(\x02\x12\x14\n\x0cv_resolution\x18\x08 \x01(\x02\x12\x16\n\x0e\x63urjudgeresult\x18\t \x01(\x05\":\n\x0f\x44\x65\x66\x65\x63tStatistic\x12\x12\n\ndefectName\x18\x01 \x01(\t\x12\x13\n\x0b\x64\x65\x66\x65\x63tCount\x18\x02 \x01(\x05\"\xca\x01\n\x0eReelDefectData\x12\x0f\n\x07reelNum\x18\x01 \x01(\t\x12\x12\n\ntotalMeter\x18\x02 \x01(\x02\x12\x12\n\nbackuptime\x18\x03 \x01(\t\x12\x13\n\x0bproductName\x18\x04 \x01(\t\x12\x0f\n\x07\x63\x65llNum\x18\x05 \x01(\x05\x12.\n\x14\x64\x65\x66\x65\x63tCountStatistic\x18\x06 \x03(\x0b\x32\x10.DefectStatistic\x12)\n\x0e\x64\x65\x66\x65\x63tDataList\x18\x07 \x03(\x0b\x32\x11.SingleDefectDatab\x06proto3')
)




_IMAGE = _descriptor.Descriptor(
  name='Image',
  full_name='Image',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='buf', full_name='Image.buf', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=24,
  serialized_end=44,
)


_FEATUREINFODATA = _descriptor.Descriptor(
  name='FeatureInfoData',
  full_name='FeatureInfoData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='featurename', full_name='FeatureInfoData.featurename', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='featureVal', full_name='FeatureInfoData.featureVal', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=46,
  serialized_end=104,
)


_SINGLEDEFECTDATA = _descriptor.Descriptor(
  name='SingleDefectData',
  full_name='SingleDefectData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='n_Id', full_name='SingleDefectData.n_Id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='meter', full_name='SingleDefectData.meter', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='defectType', full_name='SingleDefectData.defectType', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='realTimeImg', full_name='SingleDefectData.realTimeImg', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='standardImg', full_name='SingleDefectData.standardImg', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='featureData', full_name='SingleDefectData.featureData', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='h_resolution', full_name='SingleDefectData.h_resolution', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='v_resolution', full_name='SingleDefectData.v_resolution', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='curjudgeresult', full_name='SingleDefectData.curjudgeresult', index=8,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=107,
  serialized_end=339,
)


_DEFECTSTATISTIC = _descriptor.Descriptor(
  name='DefectStatistic',
  full_name='DefectStatistic',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='defectName', full_name='DefectStatistic.defectName', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='defectCount', full_name='DefectStatistic.defectCount', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=341,
  serialized_end=399,
)


_REELDEFECTDATA = _descriptor.Descriptor(
  name='ReelDefectData',
  full_name='ReelDefectData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='reelNum', full_name='ReelDefectData.reelNum', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='totalMeter', full_name='ReelDefectData.totalMeter', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='backuptime', full_name='ReelDefectData.backuptime', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='productName', full_name='ReelDefectData.productName', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cellNum', full_name='ReelDefectData.cellNum', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='defectCountStatistic', full_name='ReelDefectData.defectCountStatistic', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='defectDataList', full_name='ReelDefectData.defectDataList', index=6,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=402,
  serialized_end=604,
)

_SINGLEDEFECTDATA.fields_by_name['realTimeImg'].message_type = _IMAGE
_SINGLEDEFECTDATA.fields_by_name['standardImg'].message_type = _IMAGE
_SINGLEDEFECTDATA.fields_by_name['featureData'].message_type = _FEATUREINFODATA
_REELDEFECTDATA.fields_by_name['defectCountStatistic'].message_type = _DEFECTSTATISTIC
_REELDEFECTDATA.fields_by_name['defectDataList'].message_type = _SINGLEDEFECTDATA
DESCRIPTOR.message_types_by_name['Image'] = _IMAGE
DESCRIPTOR.message_types_by_name['FeatureInfoData'] = _FEATUREINFODATA
DESCRIPTOR.message_types_by_name['SingleDefectData'] = _SINGLEDEFECTDATA
DESCRIPTOR.message_types_by_name['DefectStatistic'] = _DEFECTSTATISTIC
DESCRIPTOR.message_types_by_name['ReelDefectData'] = _REELDEFECTDATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Image = _reflection.GeneratedProtocolMessageType('Image', (_message.Message,), dict(
  DESCRIPTOR = _IMAGE,
  __module__ = 'DefectInfoData_pb2'
  # @@protoc_insertion_point(class_scope:Image)
  ))
_sym_db.RegisterMessage(Image)

FeatureInfoData = _reflection.GeneratedProtocolMessageType('FeatureInfoData', (_message.Message,), dict(
  DESCRIPTOR = _FEATUREINFODATA,
  __module__ = 'DefectInfoData_pb2'
  # @@protoc_insertion_point(class_scope:FeatureInfoData)
  ))
_sym_db.RegisterMessage(FeatureInfoData)

SingleDefectData = _reflection.GeneratedProtocolMessageType('SingleDefectData', (_message.Message,), dict(
  DESCRIPTOR = _SINGLEDEFECTDATA,
  __module__ = 'DefectInfoData_pb2'
  # @@protoc_insertion_point(class_scope:SingleDefectData)
  ))
_sym_db.RegisterMessage(SingleDefectData)

DefectStatistic = _reflection.GeneratedProtocolMessageType('DefectStatistic', (_message.Message,), dict(
  DESCRIPTOR = _DEFECTSTATISTIC,
  __module__ = 'DefectInfoData_pb2'
  # @@protoc_insertion_point(class_scope:DefectStatistic)
  ))
_sym_db.RegisterMessage(DefectStatistic)

ReelDefectData = _reflection.GeneratedProtocolMessageType('ReelDefectData', (_message.Message,), dict(
  DESCRIPTOR = _REELDEFECTDATA,
  __module__ = 'DefectInfoData_pb2'
  # @@protoc_insertion_point(class_scope:ReelDefectData)
  ))
_sym_db.RegisterMessage(ReelDefectData)


# @@protoc_insertion_point(module_scope)
