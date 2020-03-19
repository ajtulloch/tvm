# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
if(USE_XNNPACK)
  file(GLOB XNNPACK_CONTRIB_SRC src/runtime/contrib/XNNPACK/*.cc)
  list(APPEND RUNTIME_SRCS ${XNNPACK_CONTRIB_SRC})
  include_directories(${XNNPACK_PATH}/include)
  find_library(XNNPACK_CONTRIB_LIB XNNPACK ${XNNPACK_PATH}/lib)
  find_library(XNNPACK_CONTRIB_CPUINFO_LIB cpuinfo ${XNNPACK_PATH}/lib)
  find_library(XNNPACK_CONTRIB_CLOG_LIB clog ${XNNPACK_PATH}/lib)
  find_library(XNNPACK_CONTRIB_PTHREADPOOL_LIB pthreadpool ${XNNPACK_PATH}/lib)

  list(APPEND TVM_RUNTIME_LINKER_LIBS ${XNNPACK_CONTRIB_LIB})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${XNNPACK_CONTRIB_CPUINFO_LIB})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${XNNPACK_CONTRIB_CLOG_LIB})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${XNNPACK_CONTRIB_PTHREADPOOL_LIB})

endif(USE_XNNPACK)
