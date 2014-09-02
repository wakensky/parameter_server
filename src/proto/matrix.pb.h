// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: proto/matrix.proto

#ifndef PROTOBUF_proto_2fmatrix_2eproto__INCLUDED
#define PROTOBUF_proto_2fmatrix_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 2005000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 2005000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/generated_enum_reflection.h>
#include <google/protobuf/unknown_field_set.h>
#include "proto/range.pb.h"
#include "proto/instance.pb.h"
// @@protoc_insertion_point(includes)

namespace PS {

// Internal implementation detail -- do not call these.
void  protobuf_AddDesc_proto_2fmatrix_2eproto();
void protobuf_AssignDesc_proto_2fmatrix_2eproto();
void protobuf_ShutdownFile_proto_2fmatrix_2eproto();

class MatrixInfo;

enum MatrixInfo_Type {
  MatrixInfo_Type_DENSE = 1,
  MatrixInfo_Type_SPARSE = 2,
  MatrixInfo_Type_SPARSE_BINARY = 3
};
bool MatrixInfo_Type_IsValid(int value);
const MatrixInfo_Type MatrixInfo_Type_Type_MIN = MatrixInfo_Type_DENSE;
const MatrixInfo_Type MatrixInfo_Type_Type_MAX = MatrixInfo_Type_SPARSE_BINARY;
const int MatrixInfo_Type_Type_ARRAYSIZE = MatrixInfo_Type_Type_MAX + 1;

const ::google::protobuf::EnumDescriptor* MatrixInfo_Type_descriptor();
inline const ::std::string& MatrixInfo_Type_Name(MatrixInfo_Type value) {
  return ::google::protobuf::internal::NameOfEnum(
    MatrixInfo_Type_descriptor(), value);
}
inline bool MatrixInfo_Type_Parse(
    const ::std::string& name, MatrixInfo_Type* value) {
  return ::google::protobuf::internal::ParseNamedEnum<MatrixInfo_Type>(
    MatrixInfo_Type_descriptor(), name, value);
}
// ===================================================================

class MatrixInfo : public ::google::protobuf::Message {
 public:
  MatrixInfo();
  virtual ~MatrixInfo();

  MatrixInfo(const MatrixInfo& from);

  inline MatrixInfo& operator=(const MatrixInfo& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const MatrixInfo& default_instance();

  void Swap(MatrixInfo* other);

  // implements Message ----------------------------------------------

  MatrixInfo* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const MatrixInfo& from);
  void MergeFrom(const MatrixInfo& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  typedef MatrixInfo_Type Type;
  static const Type DENSE = MatrixInfo_Type_DENSE;
  static const Type SPARSE = MatrixInfo_Type_SPARSE;
  static const Type SPARSE_BINARY = MatrixInfo_Type_SPARSE_BINARY;
  static inline bool Type_IsValid(int value) {
    return MatrixInfo_Type_IsValid(value);
  }
  static const Type Type_MIN =
    MatrixInfo_Type_Type_MIN;
  static const Type Type_MAX =
    MatrixInfo_Type_Type_MAX;
  static const int Type_ARRAYSIZE =
    MatrixInfo_Type_Type_ARRAYSIZE;
  static inline const ::google::protobuf::EnumDescriptor*
  Type_descriptor() {
    return MatrixInfo_Type_descriptor();
  }
  static inline const ::std::string& Type_Name(Type value) {
    return MatrixInfo_Type_Name(value);
  }
  static inline bool Type_Parse(const ::std::string& name,
      Type* value) {
    return MatrixInfo_Type_Parse(name, value);
  }

  // accessors -------------------------------------------------------

  // required .PS.MatrixInfo.Type type = 1;
  inline bool has_type() const;
  inline void clear_type();
  static const int kTypeFieldNumber = 1;
  inline ::PS::MatrixInfo_Type type() const;
  inline void set_type(::PS::MatrixInfo_Type value);

  // required bool row_major = 2;
  inline bool has_row_major() const;
  inline void clear_row_major();
  static const int kRowMajorFieldNumber = 2;
  inline bool row_major() const;
  inline void set_row_major(bool value);

  // optional int32 id = 3;
  inline bool has_id() const;
  inline void clear_id();
  static const int kIdFieldNumber = 3;
  inline ::google::protobuf::int32 id() const;
  inline void set_id(::google::protobuf::int32 value);

  // required .PS.PbRange row = 5;
  inline bool has_row() const;
  inline void clear_row();
  static const int kRowFieldNumber = 5;
  inline const ::PS::PbRange& row() const;
  inline ::PS::PbRange* mutable_row();
  inline ::PS::PbRange* release_row();
  inline void set_allocated_row(::PS::PbRange* row);

  // required .PS.PbRange col = 6;
  inline bool has_col() const;
  inline void clear_col();
  static const int kColFieldNumber = 6;
  inline const ::PS::PbRange& col() const;
  inline ::PS::PbRange* mutable_col();
  inline ::PS::PbRange* release_col();
  inline void set_allocated_col(::PS::PbRange* col);

  // optional uint64 nnz = 7;
  inline bool has_nnz() const;
  inline void clear_nnz();
  static const int kNnzFieldNumber = 7;
  inline ::google::protobuf::uint64 nnz() const;
  inline void set_nnz(::google::protobuf::uint64 value);

  // optional uint32 sizeof_index = 8;
  inline bool has_sizeof_index() const;
  inline void clear_sizeof_index();
  static const int kSizeofIndexFieldNumber = 8;
  inline ::google::protobuf::uint32 sizeof_index() const;
  inline void set_sizeof_index(::google::protobuf::uint32 value);

  // required uint32 sizeof_value = 9;
  inline bool has_sizeof_value() const;
  inline void clear_sizeof_value();
  static const int kSizeofValueFieldNumber = 9;
  inline ::google::protobuf::uint32 sizeof_value() const;
  inline void set_sizeof_value(::google::protobuf::uint32 value);

  // optional .PS.InstanceInfo ins_info = 13;
  inline bool has_ins_info() const;
  inline void clear_ins_info();
  static const int kInsInfoFieldNumber = 13;
  inline const ::PS::InstanceInfo& ins_info() const;
  inline ::PS::InstanceInfo* mutable_ins_info();
  inline ::PS::InstanceInfo* release_ins_info();
  inline void set_allocated_ins_info(::PS::InstanceInfo* ins_info);

  // @@protoc_insertion_point(class_scope:PS.MatrixInfo)
 private:
  inline void set_has_type();
  inline void clear_has_type();
  inline void set_has_row_major();
  inline void clear_has_row_major();
  inline void set_has_id();
  inline void clear_has_id();
  inline void set_has_row();
  inline void clear_has_row();
  inline void set_has_col();
  inline void clear_has_col();
  inline void set_has_nnz();
  inline void clear_has_nnz();
  inline void set_has_sizeof_index();
  inline void clear_has_sizeof_index();
  inline void set_has_sizeof_value();
  inline void clear_has_sizeof_value();
  inline void set_has_ins_info();
  inline void clear_has_ins_info();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  int type_;
  bool row_major_;
  ::PS::PbRange* row_;
  ::PS::PbRange* col_;
  ::google::protobuf::int32 id_;
  ::google::protobuf::uint32 sizeof_index_;
  ::google::protobuf::uint64 nnz_;
  ::PS::InstanceInfo* ins_info_;
  ::google::protobuf::uint32 sizeof_value_;

  mutable int _cached_size_;
  ::google::protobuf::uint32 _has_bits_[(9 + 31) / 32];

  friend void  protobuf_AddDesc_proto_2fmatrix_2eproto();
  friend void protobuf_AssignDesc_proto_2fmatrix_2eproto();
  friend void protobuf_ShutdownFile_proto_2fmatrix_2eproto();

  void InitAsDefaultInstance();
  static MatrixInfo* default_instance_;
};
// ===================================================================


// ===================================================================

// MatrixInfo

// required .PS.MatrixInfo.Type type = 1;
inline bool MatrixInfo::has_type() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void MatrixInfo::set_has_type() {
  _has_bits_[0] |= 0x00000001u;
}
inline void MatrixInfo::clear_has_type() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void MatrixInfo::clear_type() {
  type_ = 1;
  clear_has_type();
}
inline ::PS::MatrixInfo_Type MatrixInfo::type() const {
  return static_cast< ::PS::MatrixInfo_Type >(type_);
}
inline void MatrixInfo::set_type(::PS::MatrixInfo_Type value) {
  assert(::PS::MatrixInfo_Type_IsValid(value));
  set_has_type();
  type_ = value;
}

// required bool row_major = 2;
inline bool MatrixInfo::has_row_major() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void MatrixInfo::set_has_row_major() {
  _has_bits_[0] |= 0x00000002u;
}
inline void MatrixInfo::clear_has_row_major() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void MatrixInfo::clear_row_major() {
  row_major_ = false;
  clear_has_row_major();
}
inline bool MatrixInfo::row_major() const {
  return row_major_;
}
inline void MatrixInfo::set_row_major(bool value) {
  set_has_row_major();
  row_major_ = value;
}

// optional int32 id = 3;
inline bool MatrixInfo::has_id() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void MatrixInfo::set_has_id() {
  _has_bits_[0] |= 0x00000004u;
}
inline void MatrixInfo::clear_has_id() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void MatrixInfo::clear_id() {
  id_ = 0;
  clear_has_id();
}
inline ::google::protobuf::int32 MatrixInfo::id() const {
  return id_;
}
inline void MatrixInfo::set_id(::google::protobuf::int32 value) {
  set_has_id();
  id_ = value;
}

// required .PS.PbRange row = 5;
inline bool MatrixInfo::has_row() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void MatrixInfo::set_has_row() {
  _has_bits_[0] |= 0x00000008u;
}
inline void MatrixInfo::clear_has_row() {
  _has_bits_[0] &= ~0x00000008u;
}
inline void MatrixInfo::clear_row() {
  if (row_ != NULL) row_->::PS::PbRange::Clear();
  clear_has_row();
}
inline const ::PS::PbRange& MatrixInfo::row() const {
  return row_ != NULL ? *row_ : *default_instance_->row_;
}
inline ::PS::PbRange* MatrixInfo::mutable_row() {
  set_has_row();
  if (row_ == NULL) row_ = new ::PS::PbRange;
  return row_;
}
inline ::PS::PbRange* MatrixInfo::release_row() {
  clear_has_row();
  ::PS::PbRange* temp = row_;
  row_ = NULL;
  return temp;
}
inline void MatrixInfo::set_allocated_row(::PS::PbRange* row) {
  delete row_;
  row_ = row;
  if (row) {
    set_has_row();
  } else {
    clear_has_row();
  }
}

// required .PS.PbRange col = 6;
inline bool MatrixInfo::has_col() const {
  return (_has_bits_[0] & 0x00000010u) != 0;
}
inline void MatrixInfo::set_has_col() {
  _has_bits_[0] |= 0x00000010u;
}
inline void MatrixInfo::clear_has_col() {
  _has_bits_[0] &= ~0x00000010u;
}
inline void MatrixInfo::clear_col() {
  if (col_ != NULL) col_->::PS::PbRange::Clear();
  clear_has_col();
}
inline const ::PS::PbRange& MatrixInfo::col() const {
  return col_ != NULL ? *col_ : *default_instance_->col_;
}
inline ::PS::PbRange* MatrixInfo::mutable_col() {
  set_has_col();
  if (col_ == NULL) col_ = new ::PS::PbRange;
  return col_;
}
inline ::PS::PbRange* MatrixInfo::release_col() {
  clear_has_col();
  ::PS::PbRange* temp = col_;
  col_ = NULL;
  return temp;
}
inline void MatrixInfo::set_allocated_col(::PS::PbRange* col) {
  delete col_;
  col_ = col;
  if (col) {
    set_has_col();
  } else {
    clear_has_col();
  }
}

// optional uint64 nnz = 7;
inline bool MatrixInfo::has_nnz() const {
  return (_has_bits_[0] & 0x00000020u) != 0;
}
inline void MatrixInfo::set_has_nnz() {
  _has_bits_[0] |= 0x00000020u;
}
inline void MatrixInfo::clear_has_nnz() {
  _has_bits_[0] &= ~0x00000020u;
}
inline void MatrixInfo::clear_nnz() {
  nnz_ = GOOGLE_ULONGLONG(0);
  clear_has_nnz();
}
inline ::google::protobuf::uint64 MatrixInfo::nnz() const {
  return nnz_;
}
inline void MatrixInfo::set_nnz(::google::protobuf::uint64 value) {
  set_has_nnz();
  nnz_ = value;
}

// optional uint32 sizeof_index = 8;
inline bool MatrixInfo::has_sizeof_index() const {
  return (_has_bits_[0] & 0x00000040u) != 0;
}
inline void MatrixInfo::set_has_sizeof_index() {
  _has_bits_[0] |= 0x00000040u;
}
inline void MatrixInfo::clear_has_sizeof_index() {
  _has_bits_[0] &= ~0x00000040u;
}
inline void MatrixInfo::clear_sizeof_index() {
  sizeof_index_ = 0u;
  clear_has_sizeof_index();
}
inline ::google::protobuf::uint32 MatrixInfo::sizeof_index() const {
  return sizeof_index_;
}
inline void MatrixInfo::set_sizeof_index(::google::protobuf::uint32 value) {
  set_has_sizeof_index();
  sizeof_index_ = value;
}

// required uint32 sizeof_value = 9;
inline bool MatrixInfo::has_sizeof_value() const {
  return (_has_bits_[0] & 0x00000080u) != 0;
}
inline void MatrixInfo::set_has_sizeof_value() {
  _has_bits_[0] |= 0x00000080u;
}
inline void MatrixInfo::clear_has_sizeof_value() {
  _has_bits_[0] &= ~0x00000080u;
}
inline void MatrixInfo::clear_sizeof_value() {
  sizeof_value_ = 0u;
  clear_has_sizeof_value();
}
inline ::google::protobuf::uint32 MatrixInfo::sizeof_value() const {
  return sizeof_value_;
}
inline void MatrixInfo::set_sizeof_value(::google::protobuf::uint32 value) {
  set_has_sizeof_value();
  sizeof_value_ = value;
}

// optional .PS.InstanceInfo ins_info = 13;
inline bool MatrixInfo::has_ins_info() const {
  return (_has_bits_[0] & 0x00000100u) != 0;
}
inline void MatrixInfo::set_has_ins_info() {
  _has_bits_[0] |= 0x00000100u;
}
inline void MatrixInfo::clear_has_ins_info() {
  _has_bits_[0] &= ~0x00000100u;
}
inline void MatrixInfo::clear_ins_info() {
  if (ins_info_ != NULL) ins_info_->::PS::InstanceInfo::Clear();
  clear_has_ins_info();
}
inline const ::PS::InstanceInfo& MatrixInfo::ins_info() const {
  return ins_info_ != NULL ? *ins_info_ : *default_instance_->ins_info_;
}
inline ::PS::InstanceInfo* MatrixInfo::mutable_ins_info() {
  set_has_ins_info();
  if (ins_info_ == NULL) ins_info_ = new ::PS::InstanceInfo;
  return ins_info_;
}
inline ::PS::InstanceInfo* MatrixInfo::release_ins_info() {
  clear_has_ins_info();
  ::PS::InstanceInfo* temp = ins_info_;
  ins_info_ = NULL;
  return temp;
}
inline void MatrixInfo::set_allocated_ins_info(::PS::InstanceInfo* ins_info) {
  delete ins_info_;
  ins_info_ = ins_info;
  if (ins_info) {
    set_has_ins_info();
  } else {
    clear_has_ins_info();
  }
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace PS

#ifndef SWIG
namespace google {
namespace protobuf {

template <>
inline const EnumDescriptor* GetEnumDescriptor< ::PS::MatrixInfo_Type>() {
  return ::PS::MatrixInfo_Type_descriptor();
}

}  // namespace google
}  // namespace protobuf
#endif  // SWIG

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_proto_2fmatrix_2eproto__INCLUDED
