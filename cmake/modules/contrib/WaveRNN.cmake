message(STATUS "Build with contrib.wavernn")
file(GLOB WAVERNN_CONTRIB_SRC src/contrib/wavernn/*.cc)
list(APPEND RUNTIME_SRCS ${WAVERNN_CONTRIB_SRC})
