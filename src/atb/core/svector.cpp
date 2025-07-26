/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include "atb/svector.h"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
//! \brief 边界检查标志
constexpr bool CHECK_BOUND = true;
} // namespace

namespace atb {
MaxSizeExceeded::MaxSizeExceeded(size_t capacity, size_t size)
    : capacity_(capacity), size_(size) {}

MaxSizeExceeded::MaxSizeExceeded(size_t size)
    : capacity_(MAX_SVECTOR_SIZE), size_(size) {}

MaxSizeExceeded::~MaxSizeExceeded() noexcept {}

const char *MaxSizeExceeded::what() const noexcept {
  std::stringstream ss;
  ss << "Exceeded the max size, " << maxSize_
     << " of SVector, and got size: " << size_;
  return ss.str().c_str();
}

constexpr SVector::SVector() : size_(0) {
  for (size_t i = 0; i < DEFAULT_SVECTOR_SIZE; ++i) {
    storage_[i] = {};
  }
}

template <class T>
SVector::SVector(std::initializer_list<T> list) {
  if (CHECK_BOUND && list.size() > DEFAULT_SVECTOR_SIZE) {
    throw MaxSizeExceeded();
  }
  size_ = list.size();
  size_t i = 0;
  for (auto it = list.begin(); it != list.end() && i < size_; ++it) {
    storage_[i++] = *it;
  }
}

template <class T>
explicit SVector::SVector(size_t size, const T &value = 0) : size_(0) {
  if (CHECK_BOUND && size > DEFAULT_SVECTOR_SIZE) {
    throw MaxSizeExceeded();
  }
  size_ = size;
  for (size_t i = 0; i < size_; ++i) {
    storage_[i] = value;
  }
}

template <class T>
SVector::SVector(const SVector<T> &other) {
  if (other.heap_) {
    heap_ = reinterpret_cast<T *>(malloc(other.size_ * sizeof(T)));
    if (!heap_) {
      throw std::bad_alloc();
    }
    size_ = other.size_;
    for (size_t i = 0; i < other.size_; ++i) {
      heap_[i] = other.heap_[i];
    }
  } else {
    size_ = other.size_;
    for (size_t i = 0; i < other.size_; ++i) {
      storage_[i] = other.storage_[i];
    }
  }
}

~SVector() {
  if (heap_) {
    free(heap_);
  }
}

template <class T>
void SVector::push_back(const T &val) noexcept(
    (!CHECK_BOUND) && std::is_nothrow_assignable<T, const T &>::value) {
  if (heap_) {
    if (CHECK_BOUND && size_ == capacity_) {
      throw MaxSizeExceeded();
    }
    heap_[size_++] = val;
    return;
  }
  if (CHECK_BOUND && size_ == DEFAULT_SVECTOR_SIZE) {
    throw MaxSizeExceeded();
  }
  storage_[size_++] = val;
}

template <class T>
T *SVector::begin() noexcept {
  if (heap_) {
    return &heap_[0];
  }
  return &storage_[0];
}

template <class T>
const T *SVector::begin() const noexcept {
  if (heap_) {
    return &heap_[0];
  }
  return &storage_[0];
}

template <class T>
T *SVector::end() noexcept {
  if (heap_) {
    return (&heap_[0]) + size_;
  }
  return (&storage_[0]) + size_;
}

template <class T>
const T *SVector::end() const noexcept {
  if (heap_) {
    return (&heap_[0]) + size_;
  }
  return (&storage_[0]) + size_;
}

template <class T>
T &SVector::operator[](size_t i) {
  if (heap_) {
    if (size_ == 0 || i >= size_) {
      throw std::out_of_range("out of range");
    }
    return heap_[i];
  }
  if (size_ == 0 || i >= size_) {
    throw std::out_of_range("out of range");
  }
  return storage_[i];
}

template <class T>
const T &SVector::operator[](size_t i) const {
  if (heap_) {
    if (size_ == 0 || i >= size_) {
      throw std::out_of_range("out of range");
    }
    return heap_[i];
  }
  if (size_ == 0 || i >= size_) {
    throw std::out_of_range("out of range");
  }
  return storage_[i];
}

template <class T>
T &SVector::at(size_t i) {
  if (heap_) {
    if (size_ == 0 || i >= size_) {
      throw std::out_of_range("out of range");
    }
    return heap_[i];
  } else {
    if (size_ == 0 || i >= size_ || i > DEFAULT_SVECTOR_SIZE) {
      throw std::out_of_range("out of range");
    }
    return storage_[i];
  }
}

template <class T>
const T &SVector::at(size_t i) const {
  if (heap_) {
    if (size_ == 0 || i >= size_) {
      throw std::out_of_range("heap out of range");
    }
    return heap_[i];
  }
  if (size_ == 0 || i >= size_ || i > DEFAULT_SVECTOR_SIZE) {
    throw std::out_of_range("stack out of range");
  }
  return storage_[i];
}

template <class T>
size_t SVector::size() const noexcept { return size_; }

template <class T>
void SVector::insert(const size_t pos, const T &value) noexcept(
    (!CHECK_BOUND) && std::is_nothrow_assignable<T, const T &>::value) {
  if (heap_) {
    if (pos > size_ || pos == capacity_) {
      throw MaxSizeExceeded();
    }
    for (auto it = size_; it != pos; it--) {
      heap_[it] = heap_[it - 1];
    }
    heap_[pos] = value;
    size_ += 1;
    return;
  }
  if (CHECK_BOUND && size_ == DEFAULT_SVECTOR_SIZE) {
    throw MaxSizeExceeded();
  }
  if (pos > size_) {
    throw MaxSizeExceeded();
  }

  for (auto it = size_; it != pos; it--) {
    storage_[it] = storage_[it - 1];
  }
  storage_[pos] = value;
  size_ += 1;
  return;
}

template <class T>
bool SVector::empty() const noexcept { return size_ == 0; }

void SVector::clear() noexcept { size_ = 0; }

template <class T>
T *SVector::data() noexcept {
  if (heap_) {
    return &heap_[0];
  }
  return &storage_[0];
}

template <class T>
const T *SVector::data() const noexcept {
  if (heap_) {
    return &heap_[0];
  }
  return &storage_[0];
}

template <class T>
void SVector::resize(size_t size) {
  if (heap_ && size > capacity_) {
    throw MaxSizeExceeded();
  }
  size_ = size;
}

template <class T>
void SVector::reserve(size_t size) {
  if (size > MAX_SVECTOR_SIZE) {
    throw MaxSizeExceeded();
  }

  if (size > DEFAULT_SVECTOR_SIZE) {
    if (heap_) {
      free(heap_);
    }
    heap_ = reinterpret_cast<T *>(malloc(size * sizeof(T)));
    if (!heap_) {
      throw std::bad_alloc();
    }
    for (size_t i = 0; i < size; ++i) {
      heap_[i] = {};
    }
    capacity_ = size;
  }
}

template <class T>
bool SVector::operator==(const SVector<T> &other) const {
  if (heap_) {
    if (size_ != other.size_ || !other.heap_) {
      return false;
    }
    for (size_t i = 0; i < size_; ++i) {
      if (heap_[i] != other.heap_[i]) {
        return false;
      }
    }
  } else {
    if (size_ != other.size_) {
      return false;
    }
    for (size_t i = 0; i < size_; ++i) {
      if (storage_[i] != other.storage_[i]) {
        return false;
      }
    }
  }
  return true;
}

template <class T>
bool SVector::operator!=(const SVector<T> &other) const {
  if (heap_) {
    if (size_ != other.size_ || !other.heap_) {
      return true;
    }
    for (size_t i = 0; i < size_; ++i) {
      if (heap_[i] != other.heap_[i]) {
        return true;
      }
    }
  } else {
    if (size_ != other.size_) {
      return true;
    }
    for (size_t i = 0; i < size_; ++i) {
      if (storage_[i] != other.storage_[i]) {
        return true;
      }
    }
  }
  return false;
}

template <class T>
bool SVector::operator<(const SVector<T> &other) const {
  if (heap_) {
    if (size_ != other.size_ || !other.heap_) {
      return size_ < other.size_;
    }
    for (size_t i = 0; i < size_; ++i) {
      if (heap_[i] != other.heap_[i]) {
        return heap_[i] < other.heap_[i];
      }
    }
  } else {
    if (size_ != other.size_) {
      return size_ < other.size_;
    }
    for (size_t i = 0; i < size_; ++i) {
      if (storage_[i] != other.storage_[i]) {
        return storage_[i] < other.storage_[i];
      }
    }
  }
  return false;
}

template <class T>
SVector &SVector::operator=(std::initializer_list<T> list) {
  if (heap_) {
    if (CHECK_BOUND && list.size() > MAX_SVECTOR_SIZE) {
      throw MaxSizeExceeded();
    }
    size_ = list.size();
    size_t i = 0;
    for (auto it = list.begin(); it != list.end() && i < size_; ++it) {
      heap_[i++] = *it;
    }
    return *this;
  } else {
    if (CHECK_BOUND && list.size() > DEFAULT_SVECTOR_SIZE) {
      throw MaxSizeExceeded();
    }
    size_ = list.size();
    size_t i = 0;
    for (auto it = list.begin(); it != list.end() && i < size_; ++it) {
      storage_[i++] = *it;
    }
    return *this;
  }
}

template <class T>
SVector &SVector::operator=(const SVector &other) {
  if (heap_) {
    size_ = other.size_;
    for (size_t i = 0; i < other.size_; ++i) {
      heap_[i] = other.heap_[i];
    }
    return *this;
  } else {
    size_ = other.size_;
    for (size_t i = 0; i < other.size_; ++i) {
      storage_[i] = other.storage_[i];
    }
    return *this;
  }
}

template <class T>
std::ostream &operator<<(std::ostream &os, const SVector<T> &svector) {
  if (svector.size() == 0) {
    return os;
  }

  std::string str = ",";
  for (size_t i = 0; i < svector.size(); ++i) {
    os << svector.at(i);
    if (i != svector.size() - 1) {
      os << str;
    }
  }

  return os;
}
} // namespace atb
