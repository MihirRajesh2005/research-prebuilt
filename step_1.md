Yep. Let's do **Step 1 as a concrete build milestone**, not as a tour of ONNX Runtime's documentation.

Your goal for Step 1 is:

> **Build a tiny C++/Clang program with CMake that loads an ONNX model and runs it on your RTX 5060 through ONNX Runtime's CUDA Execution Provider.**

Don't integrate the ray tracer, OIDN, EXR, quantization, or Nsight yet. If we get this working, the hardest dependency in the eventual C++ pipeline is proven.

ONNX Runtime officially supports a C/C++ API and a CUDA Execution Provider for NVIDIA GPUs. ([ONNX Runtime][1])

## 0. What we're going to build

Eventually:

```text
C++ / Clang
    │
    ▼
CMake project
    │
    ▼
ONNX Runtime
    │
    ▼
CUDA Execution Provider
    │
    ▼
RTX 5060
    │
    ▼
ONNX model
```

For now, though, we'll use **any small known-good ONNX model**. We don't want OIDN's model conversion to complicate our first test.

---

# 1. Check your existing environment

Since you're on Windows 11, open a terminal and check:

```powershell
clang++ --version
cmake --version
nvcc --version
nvidia-smi
```

You should have:

* Clang
* CMake
* CUDA Toolkit
* NVIDIA driver exposing your RTX 5060

The current ONNX Runtime documentation recommends CMake 3.28+ when building ONNX Runtime itself, although **we don't need to build ONNX Runtime from source just to use it**. ([ONNX Runtime][2])

### Important distinction

We are **not building ONNX Runtime**.

We're consuming a prebuilt ONNX Runtime distribution.

That saves us a considerable amount of unnecessary work.

---

# 2. Install the CUDA-enabled ONNX Runtime

For your C++ application, you need the **GPU/CUDA build**, not the CPU-only package.

ONNX Runtime provides GPU distributions for Windows and Linux, and its CUDA Execution Provider is specifically what lets the runtime execute ONNX operators on an NVIDIA GPU. ([ONNX Runtime][1])

There are several ways to obtain it. For this project, I'd avoid making NuGet or Visual Studio the centre of the build system since you specifically want **CMake + Clang**.

Download the appropriate **ONNX Runtime GPU release package** from the official releases.

You'll end up with something conceptually like:

```text
onnxruntime-gpu/
├── include/
├── lib/
│   └── onnxruntime.lib
└── bin/
    └── onnxruntime.dll
```

The exact contents can vary by release.

The CUDA EP also has version requirements involving CUDA and cuDNN, so **don't blindly grab an arbitrary old ONNX Runtime release**. The current compatibility table is maintained in the CUDA EP documentation. ([ONNX Runtime][3])

---

# 3. Make the project

I'd start with:

```text
oidn-quant/
│
├── CMakeLists.txt
├── src/
│   └── main.cpp
│
└── third_party/
    └── onnxruntime/
```

Put the ONNX Runtime distribution under:

```text
third_party/onnxruntime/
```

so that your project becomes self-contained.

For example:

```text
third_party/onnxruntime/
├── include/
├── lib/
└── bin/
```

---

# 4. Your first CMakeLists.txt

Don't make this complicated yet.

Something along these lines:

```cmake
cmake_minimum_required(VERSION 3.28)

project(oidn_quant
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_ROOT
    "${CMAKE_SOURCE_DIR}/third_party/onnxruntime"
)

add_executable(benchmark
    src/main.cpp
)

target_include_directories(benchmark PRIVATE
    "${ONNXRUNTIME_ROOT}/include"
)

target_link_libraries(benchmark PRIVATE
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
)

add_custom_command(TARGET benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/bin/onnxruntime.dll"
        "$<TARGET_FILE_DIR:benchmark>"
)
```

This is deliberately boring.

That's good.

We're establishing:

```text
CMake
 ↓
Clang
 ↓
C++
 ↓
ONNX Runtime
```

before adding anything else.

---

# 5. Write the smallest possible C++ program

Your first `main.cpp` should **not even run inference yet**.

Just prove that the C++ API is accessible:

```cpp
#include <iostream>

#include <onnxruntime_cxx_api.h>

int main()
{
    Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "oidn-quant"
    );

    std::cout << "ONNX Runtime initialized.\n";

    return 0;
}
```

This uses ONNX Runtime's C++ wrapper around its C API.

The official C API workflow is essentially:

1. create an environment,
2. create session options,
3. create a session,
4. create tensors,
5. call `OrtRun`.

The C++ API wraps these concepts in `Ort::` classes. ([ONNX Runtime][1])

---

# 6. Configure with Clang

From the project root:

```powershell
cmake -S . -B build -G Ninja `
    -DCMAKE_CXX_COMPILER=clang++
```

Then:

```powershell
cmake --build build --config Release
```

If everything works:

```powershell
.\build\benchmark.exe
```

should print:

```text
ONNX Runtime initialized.
```

**Stop here if this doesn't work.**

Don't start debugging CUDA, ONNX models, and OIDN simultaneously. First get the library linked.

---

# 7. Then enable CUDA

Once the CPU-side initialization works, modify the program.

The important concept is the **CUDA Execution Provider**.

ONNX Runtime's C API provides `OrtSessionOptionsAppendExecutionProvider_CUDA`, and the CUDA EP documentation provides C/C++ examples for registering it. ([ONNX Runtime][1])

Conceptually:

```cpp
Ort::SessionOptions session_options;

session_options.AppendExecutionProvider_CUDA(0);
```

Then:

```cpp
Ort::Session session(
    env,
    L"model.onnx",
    session_options
);
```

The `0` means GPU 0.

For your machine, that's your RTX 5060.

---

# 8. One wrinkle with the C++ API

Depending on the exact ONNX Runtime release you're using, the convenience C++ CUDA API may differ slightly from examples you find online.

That's why I'd use the **documentation corresponding to your installed ORT version**, rather than copying a random Stack Overflow/GitHub snippet.

The current CUDA EP documentation provides both legacy and newer provider-options APIs. ([ONNX Runtime][3])

If the simple:

```cpp
session_options.AppendExecutionProvider_CUDA(0);
```

doesn't compile, **don't start changing random headers**. Tell me the ORT version and compiler error and we'll use the appropriate API for that version.

---

# 9. Then load an ONNX model

Once CUDA initialization works, the next milestone is:

```text
C++
 │
 ├── Ort::Env
 ├── Ort::SessionOptions
 │      └── CUDA EP
 │
 └── Ort::Session
         │
         ▼
      model.onnx
```

At this point you should be able to query:

```cpp
session.GetInputCount();
session.GetOutputCount();
```

and inspect the input/output names and tensor dimensions.

That is important because **you shouldn't hardcode assumptions about OIDN's model input layout until we've inspected the actual exported model.**

---

# 10. Don't worry about tensor data yet

The next step after that will be creating an `Ort::Value` containing your input tensor.

Eventually you'll have something like:

```text
EXR
 │
 ▼
float buffer
 │
 ▼
Ort::Value tensor
 │
 ▼
session.Run()
 │
 ▼
Ort::Value output
 │
 ▼
EXR
```

But that's **Step 1.5**, not Step 1.

For now I want to establish:

### Milestone A

```text
CMake + Clang
        ↓
C++ executable
        ↓
ONNX Runtime
        ↓
CUDA EP
        ↓
RTX 5060
```

Once that works, we know your fundamental C++ inference environment is sound.

---

## One thing I would change from my earlier advice

I previously suggested that you might want to build with CMake and Clang while using ONNX Runtime's prebuilt binaries. **I still recommend that.**

Do **not** build ONNX Runtime from source unless we discover that OIDN's model or your profiling requirements require it.

ONNX Runtime's source build has a substantial dependency/toolchain setup, whereas the project explicitly provides prebuilt GPU packages. ([ONNX Runtime][3])

There's also a potentially useful future optimization: ONNX Runtime can be built with NVTX profiling support so Nsight Systems can correlate CUDA kernels with their ONNX operators, but **we don't need to touch that yet**. ([ONNX Runtime][4])

### So your immediate task is just this:

```text
1. Check clang / CMake / CUDA / driver
2. Obtain matching ONNX Runtime GPU package
3. Create CMake project
4. Link ONNX Runtime
5. Run Ort::Env
6. Enable CUDA EP
7. Confirm a session can be created on GPU
```

**Once you've got steps 1–6 working, stop there and tell me what you've got.** Then we'll do the next piece: loading an ONNX model and actually executing a tensor through your RTX 5060.

[1]: https://onnxruntime.ai/docs/get-started/with-c.html?utm_source=chatgpt.com "C | onnxruntime"
[2]: https://onnxruntime.ai/docs/build/inferencing.html?utm_source=chatgpt.com "Build for inferencing | onnxruntime"
[3]: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html?utm_source=chatgpt.com "NVIDIA - CUDA | onnxruntime"
[4]: https://onnxruntime.ai/docs/build/eps.html?utm_source=chatgpt.com "Build with different EPs | onnxruntime"
