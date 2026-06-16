---
title: 一次简单的Unity游戏逆向以及金手指制作
published: 2026-06-16
description: '使用Kiero2+MinHook+Imgui+UnityResolve'
image: ''
tags: [Unity,Reverse]
category: 'Unity'
draft: false 
lang: 'zh_CN'
---
闲得无聊找到一款Unity制作的游戏，因游戏设计问题导致了卡关，又非常想体验后续的关卡情节，因此无奈只能制作一个金手指。

**目标：修改/锁定玩家的 Mental 值**

首先需要知道Unity一共有两种脚本后处理方式(将编写的脚本转换为可执行代码):`Mono`和`IL2CPP`。在此不做过多介绍。

查看游戏文件夹，可以发现游戏只有`mono-2.0-bdwgc.dll`，没有il2cpp的特征:`global-metadata.dat`,可以初步判断这个游戏仅使用了Mono。

这就很简单了,由于 Mono编译利用的是`.NET`实现，我们可以直接利用现成的[dnSpyEx](https://github.com/dnSpyEx/dnSpy)来反编译游戏的主要功能实现模块`Assembly-CSharp.dll`，来获取游戏具体信息。

直接将dll拖入dnSpy,搜索有关玩家的关键词，可以确定我们要修改的值就是

```csharp Assembly-CSharp.dll
// PlayerData
// Token: 0x0400029E RID: 670
[Header("Stats")]
[Tooltip("Current mental health value")]
public float currentMental = 100f;
```

由于此游戏并未加密，我们可以直接查看到函数的内部逻辑，这样我们就有了两条路可以走：

- 1.直接利用dnSpy修改游戏中SetMental函数，直接跳过逻辑。
- 2.不修改游戏本体，用其它方式获取修改此值。

**第一条路:**

找到函数：

```csharp Assembly-CSharp.dll
// PlayerData
// Token: 0x060001A9 RID: 425 RVA: 0x0000CA28 File Offset: 0x0000AC28
public void SetMental(float value)
{
 this.currentMental = Mathf.Clamp(value, 0f, this.maxMental);
}
```

右键，选中`编辑方法`。
我直接用了最简单的方式。

```csharp Assembly-CSharp.dll
// PlayerData
// Token: 0x060001F9 RID: 505 RVA: 0x00004C82 File Offset: 0x00002E82
public void SetMental(float value)
{
 this.currentMental = 100f;
}
```

修改完成之后，在左上角`文件`中选择`保存模块`，确定之后即修改成功。(记得先备份游戏原dll)。

**第二条路:**

这里我选择的方案是传统的`version.dll`劫持法，(简单来说Windows的dll加载路径优先级是本地路径高于Windows系统目录下的路径，因此我们在本地放入一个和系统dll同名的dll即可让程序加载我们自己的dll。但是需要做好函数转发，不然当程序调用原dll的函数的时候会发现没有此函数或功能不正常而报错。)
直接用[AheadLibEx](https://github.com/i1tao/AheadLibEx)生成直接方案，将生成出来的version.dll放进游戏文件夹测试：

```c version.dll
DWORD WINAPI patch_thread_proc(LPVOID context)
{
    UNREFERENCED_PARAMETER(context);
    MessageBoxA(NULL, "AheadLibEx", "AheadLibEx", MB_ICONINFORMATION);
    return 0;
}
```

可以发现成功了，但是却弹了两个MessageBox，为什么只开了一次游戏，但是弹了两个呢？

修改代码，打印出当前进程的名字：

```c version.dll
DWORD WINAPI patch_thread_proc(LPVOID context)
{
    UNREFERENCED_PARAMETER(context);
    CHAR process_path[MAX_PATH] = { 0 };
    CHAR* process_name = NULL;
    CHAR message[MAX_PATH * 2] = { 0 };
    GetModuleFileNameA(NULL, process_path, MAX_PATH);
    process_name = strrchr(process_path, '\\');
    process_name = (process_name != NULL) ? (process_name + 1) : process_path;
    wsprintfA(message, "Current process: %s", process_name);
    MessageBoxA(NULL, message, "AheadLibEx", MB_ICONINFORMATION);
    return 0;
}
```

可以发现，除了游戏本体外，同时打开了`UnityCrashHandler64.exe`也加载了我们的dll。因此我们只需要判断一下当前进程名，只在当游戏本体加载的时候进行下一步操作。

```c version.dll
if (process_name && _stricmp(process_name, "unitygame.exe") == 0)
{
    Main();
}
```

这里还有一个容易漏掉的点：`version.dll` 不是只放一个入口函数就完事。游戏原本如果调用系统 `version.dll` 导出的函数，我们的同名 DLL 也必须把这些导出转发给真正的系统 DLL。AheadLibEx 生成的工程会包含这部分转发代码，我们只在自己的初始化线程里做 Hook 和菜单逻辑，系统 API 的实际功能仍然交给系统目录下的 `version.dll`。

入口处也不要在 `DllMain` 里直接做大量初始化。这里用 `CreateThread` 开一个线程去跑 `Main()`，并且只在目标游戏进程里启动。

```cpp dllmain.cpp
DWORD WINAPI MainThread(LPVOID)
{
    Main();
    return 0;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID)
{
    if (reason != DLL_PROCESS_ATTACH)
        return TRUE;

    DisableThreadLibraryCalls(hModule);

    wchar_t path[MAX_PATH]{};
    if (GetModuleFileNameW(nullptr, path, MAX_PATH) == 0)
        return FALSE;

    const wchar_t* exeName = wcsrchr(path, L'\\');
    exeName = exeName ? exeName + 1 : path;

    if (wcscmp(exeName, L"unitygame.exe") == 0)
    {
        HANDLE thread = CreateThread(nullptr, 0, MainThread, nullptr, 0, nullptr);
        if (thread)
            CloseHandle(thread);
    }

    return TRUE;
}
```

这里我选择使用 ImGui 作为图形化界面。由于 ImGui 并不属于 Unity 原本的渲染流程，如果希望它能够稳定地显示在游戏画面上，并且与游戏帧同步刷新，就需要 Hook Unity 底层所使用的图形 API 渲染函数，通过在每一帧画面提交前后插入 ImGui 的绘制逻辑，可以让 ImGui 与游戏画面一起完成渲染。当然，也可以不 Hook 渲染函数，而是创建一个独立的透明窗口作为覆盖层，但这种方式在窗口同步、焦点切换和输入处理上会更麻烦，因此这里先采用 Hook 渲染流程的方式。

这里我直接选用了[kiero2](https://github.com/kirchesz/kiero2)库，这个库可以快速定位到渲染后端的上下文。
翻阅游戏文件夹可知此游戏用的是D3D12渲染，下载 kiero 后，把 `kiero.hpp`、`kiero_d3d12.hpp`、`kiero_intern.hpp`、`kiero_intern.cpp`、`kiero_d3d12.cpp` 加入项目，再把 `imgui.cpp`、`imgui_draw.cpp`、`imgui_tables.cpp`、`imgui_widgets.cpp`、`backends/imgui_impl_dx12.cpp`、`backends/imgui_impl_win32.cpp` 加入项目。

核心思路是：kiero 负责拿到 D3D12 相关对象的虚表地址，MinHook 负责挂钩；`Present1` 每帧绘制 ImGui，`ExecuteCommandLists` 捕获 Unity 实际使用的 `ID3D12CommandQueue`，`ResizeBuffers` 在窗口尺寸变化后重建 RTV。下面的 `13`、`22`、`10` 是对应 COM 虚表中的方法下标，分别对应 `ResizeBuffers`、`Present1`、`ExecuteCommandLists`。

```cpp dllmain.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(
    HWND hWnd,
    UINT msg,
    WPARAM wParam,
    LPARAM lParam);

constexpr std::size_t kSwapChainResizeBuffersIndex = 13;
constexpr std::size_t kSwapChainPresent1Index = 22;
constexpr std::size_t kCommandQueueExecuteCommandListsIndex = 10;

using ResizeBuffersFn = HRESULT(STDMETHODCALLTYPE*)(
    IDXGISwapChain* swapChain,
    UINT bufferCount,
    UINT width,
    UINT height,
    DXGI_FORMAT newFormat,
    UINT swapChainFlags);

using Present1Fn = HRESULT(STDMETHODCALLTYPE*)(
    IDXGISwapChain1* swapChain,
    UINT syncInterval,
    UINT flags,
    const DXGI_PRESENT_PARAMETERS* presentParameters);

using ExecuteCommandListsFn = void(STDMETHODCALLTYPE*)(
    ID3D12CommandQueue* commandQueue,
    UINT numCommandLists,
    ID3D12CommandList* const* commandLists);

struct FrameContext
{
    ID3D12CommandAllocator* commandAllocator = nullptr;
    ID3D12Resource* renderTarget = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE renderTargetDescriptor{};
};

ResizeBuffersFn g_originalResizeBuffers = nullptr;
Present1Fn g_originalPresent1 = nullptr;
ExecuteCommandListsFn g_originalExecuteCommandLists = nullptr;

HWND g_hwnd = nullptr;
WNDPROC g_originalWndProc = nullptr;
ID3D12Device* g_device = nullptr;
ID3D12CommandQueue* g_commandQueue = nullptr;
ID3D12GraphicsCommandList* g_commandList = nullptr;
ID3D12DescriptorHeap* g_rtvHeap = nullptr;
ID3D12DescriptorHeap* g_srvHeap = nullptr;
FrameContext* g_frameContexts = nullptr;
UINT g_bufferCount = 0;
UINT g_frameIndex = 0;
bool g_imguiInitialized = false;
bool g_menuVisible = true;
std::atomic_bool g_lockMental = false;
```

RTV、命令分配器和命令列表按 SwapChain 的后缓冲数量创建：

```cpp dllmain.cpp
bool CreateRenderTargets(IDXGISwapChain* swapChain)
{
    DXGI_SWAP_CHAIN_DESC swapChainDesc{};
    if (FAILED(swapChain->GetDesc(&swapChainDesc)))
        return false;

    g_hwnd = swapChainDesc.OutputWindow;
    g_bufferCount = swapChainDesc.BufferCount;
    if (g_bufferCount == 0)
        return false;

    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc{};
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.NumDescriptors = g_bufferCount;
    rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    rtvHeapDesc.NodeMask = 1;

    if (FAILED(g_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&g_rtvHeap))))
        return false;

    const UINT rtvDescriptorSize =
        g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = g_rtvHeap->GetCPUDescriptorHandleForHeapStart();

    g_frameContexts = new FrameContext[g_bufferCount];
    for (UINT i = 0; i < g_bufferCount; ++i)
    {
        if (FAILED(g_device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&g_frameContexts[i].commandAllocator))))
            return false;

        if (FAILED(swapChain->GetBuffer(i, IID_PPV_ARGS(&g_frameContexts[i].renderTarget))))
            return false;

        g_frameContexts[i].renderTargetDescriptor = rtvHandle;
        g_device->CreateRenderTargetView(g_frameContexts[i].renderTarget, nullptr, rtvHandle);
        rtvHandle.ptr += rtvDescriptorSize;
    }

    if (FAILED(g_device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        g_frameContexts[0].commandAllocator,
        nullptr,
        IID_PPV_ARGS(&g_commandList))))
        return false;

    g_commandList->Close();
    return true;
}
```

初始化 ImGui 时，`ImGui_ImplDX12_InitInfo` 里要传入从 `ExecuteCommandLists` 捕获到的 `g_commandQueue`：

```cpp dllmain.cpp
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (g_imguiInitialized &&
        g_menuVisible &&
        ImGui_ImplWin32_WndProcHandler(hwnd, msg, wParam, lParam))
        return TRUE;

    return CallWindowProcW(g_originalWndProc, hwnd, msg, wParam, lParam);
}

bool InitializeImGui(IDXGISwapChain* swapChain)
{
    if (g_imguiInitialized)
        return true;

    if (!g_commandQueue)
        return false;

    if (FAILED(swapChain->GetDevice(IID_PPV_ARGS(&g_device))))
        return false;

    DXGI_SWAP_CHAIN_DESC swapChainDesc{};
    if (FAILED(swapChain->GetDesc(&swapChainDesc)))
        return false;

    D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc{};
    srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvHeapDesc.NumDescriptors = 1;
    srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    if (FAILED(g_device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&g_srvHeap))))
        return false;

    if (!CreateRenderTargets(swapChain))
        return false;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    if (!ImGui_ImplWin32_Init(g_hwnd))
        return false;

    ImGui_ImplDX12_InitInfo initInfo{};
    initInfo.Device = g_device;
    initInfo.CommandQueue = g_commandQueue;
    initInfo.NumFramesInFlight = static_cast<int>(g_bufferCount);
    initInfo.RTVFormat = swapChainDesc.BufferDesc.Format;
    initInfo.DSVFormat = DXGI_FORMAT_UNKNOWN;
    initInfo.SrvDescriptorHeap = g_srvHeap;
    initInfo.LegacySingleSrvCpuDescriptor = g_srvHeap->GetCPUDescriptorHandleForHeapStart();
    initInfo.LegacySingleSrvGpuDescriptor = g_srvHeap->GetGPUDescriptorHandleForHeapStart();

    if (!ImGui_ImplDX12_Init(&initInfo))
        return false;

    g_originalWndProc = reinterpret_cast<WNDPROC>(
        SetWindowLongPtrW(g_hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(WndProc)));

    g_imguiInitialized = true;
    return true;
}
```

菜单渲染单独拆出来写。`ImGui::Checkbox` 需要传入 `bool*`，而开关变量使用 `std::atomic_bool`，所以这里先 `load()` 到局部变量，用户点击后再 `store()` 回去：

```cpp dllmain.cpp
void UpdateMenuVisible()
{
    static bool insertPressedLastFrame = false;
    const bool insertPressed = (GetAsyncKeyState(VK_INSERT) & 0x8000) != 0;

    if (insertPressed && !insertPressedLastFrame)
        g_menuVisible = !g_menuVisible;

    insertPressedLastFrame = insertPressed;
}

void RenderMenu()
{
    UpdateMenuVisible();
    if (!g_menuVisible)
        return;

    ImGui::SetNextWindowSize(ImVec2(360.0f, 0.0f), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("xiallo", &g_menuVisible, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::End();
        return;
    }

    ImGui::Text("D3D12 ImGui hook");
    ImGui::Separator();

    const bool setMentalReady = InitializeSetMentalHooks();
    bool lockMental = g_lockMental.load();
    if (ImGui::Checkbox("LockMental", &lockMental))
        g_lockMental.store(lockMental);
    ImGui::SameLine();
    ImGui::TextDisabled(setMentalReady ? "ready" : "waiting");

    ImGui::End();
}
```

每帧绘制时，把后缓冲从 `D3D12_RESOURCE_STATE_PRESENT` 切到 `D3D12_RESOURCE_STATE_RENDER_TARGET`，画完再切回去：

```cpp dllmain.cpp
void RenderImGui(IDXGISwapChain* swapChain)
{
    IDXGISwapChain3* swapChain3 = nullptr;
    UINT bufferIndex = 0;

    if (SUCCEEDED(swapChain->QueryInterface(IID_PPV_ARGS(&swapChain3))))
    {
        bufferIndex = swapChain3->GetCurrentBackBufferIndex();
        swapChain3->Release();
    }
    else if (g_bufferCount > 0)
    {
        bufferIndex = g_frameIndex % g_bufferCount;
        ++g_frameIndex;
    }

    FrameContext& frameContext = g_frameContexts[bufferIndex];
    frameContext.commandAllocator->Reset();
    g_commandList->Reset(frameContext.commandAllocator, nullptr);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = frameContext.renderTarget;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    g_commandList->ResourceBarrier(1, &barrier);

    g_commandList->OMSetRenderTargets(1, &frameContext.renderTargetDescriptor, FALSE, nullptr);
    g_commandList->SetDescriptorHeaps(1, &g_srvHeap);

    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    RenderMenu();

    ImGui::Render();
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), g_commandList);

    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    g_commandList->ResourceBarrier(1, &barrier);
    g_commandList->Close();

    ID3D12CommandList* commandLists[] = { g_commandList };
    g_commandQueue->ExecuteCommandLists(1, commandLists);
}
```

三个 Hook 函数：

```cpp dllmain.cpp
void ReleaseRenderTargets()
{
    if (g_frameContexts)
    {
        for (UINT i = 0; i < g_bufferCount; ++i)
        {
            if (g_frameContexts[i].renderTarget)
                g_frameContexts[i].renderTarget->Release();

            if (g_frameContexts[i].commandAllocator)
                g_frameContexts[i].commandAllocator->Release();
        }

        delete[] g_frameContexts;
        g_frameContexts = nullptr;
    }

    if (g_rtvHeap)
    {
        g_rtvHeap->Release();
        g_rtvHeap = nullptr;
    }

    g_bufferCount = 0;
    g_frameIndex = 0;
}

HRESULT STDMETHODCALLTYPE HookPresent1(
    IDXGISwapChain1* swapChain,
    UINT syncInterval,
    UINT flags,
    const DXGI_PRESENT_PARAMETERS* presentParameters)
{
    if (InitializeImGui(swapChain))
        RenderImGui(swapChain);

    return g_originalPresent1(swapChain, syncInterval, flags, presentParameters);
}

HRESULT STDMETHODCALLTYPE HookResizeBuffers(
    IDXGISwapChain* swapChain,
    UINT bufferCount,
    UINT width,
    UINT height,
    DXGI_FORMAT newFormat,
    UINT swapChainFlags)
{
    ReleaseRenderTargets();

    const HRESULT result =
        g_originalResizeBuffers(swapChain, bufferCount, width, height, newFormat, swapChainFlags);

    if (SUCCEEDED(result) && g_device)
        CreateRenderTargets(swapChain);

    return result;
}

void STDMETHODCALLTYPE HookExecuteCommandLists(
    ID3D12CommandQueue* commandQueue,
    UINT numCommandLists,
    ID3D12CommandList* const* commandLists)
{
    if (g_commandQueue != commandQueue)
    {
        if (g_commandQueue)
            g_commandQueue->Release();

        g_commandQueue = commandQueue;
        g_commandQueue->AddRef();
    }

    g_originalExecuteCommandLists(commandQueue, numCommandLists, commandLists);
}
```

最后在初始化线程里等待 `dxgi.dll` 和 `d3d12.dll` 加载，使用 kiero 定位 D3D12 虚表，再用 MinHook 挂上面三个函数：

```cpp dllmain.cpp
void Main()
{
    while (!GetModuleHandleW(L"dxgi.dll") || !GetModuleHandleW(L"d3d12.dll"))
        Sleep(100);

    kiero::D3D12Output d3d12{};
    const kiero::Error err = kiero::locate<kiero::Implementation_D3D12>(nullptr, &d3d12);
    if (err != kiero::Error_Nil)
        return;

    if (d3d12.swapchain_methods.size() <= kSwapChainPresent1Index ||
        d3d12.command_queue_methods.size() <= kCommandQueueExecuteCommandListsIndex)
        return;

    if (MH_Initialize() != MH_OK)
        return;

    if (MH_CreateHook(
        d3d12.swapchain_methods[kSwapChainResizeBuffersIndex],
        &HookResizeBuffers,
        reinterpret_cast<LPVOID*>(&g_originalResizeBuffers)) != MH_OK)
        return;

    if (MH_CreateHook(
        d3d12.swapchain_methods[kSwapChainPresent1Index],
        &HookPresent1,
        reinterpret_cast<LPVOID*>(&g_originalPresent1)) != MH_OK)
        return;

    if (MH_CreateHook(
        d3d12.command_queue_methods[kCommandQueueExecuteCommandListsIndex],
        &HookExecuteCommandLists,
        reinterpret_cast<LPVOID*>(&g_originalExecuteCommandLists)) != MH_OK)
        return;

    MH_EnableHook(MH_ALL_HOOKS);
}
```

渲染imgui菜单之后，我们开始着手于功能的编写。这里我们用现成的[UnityResolve](https://github.com/1992724048/UnityResolve.hpp)库。

UnityResolve 的作用可以简单理解为：在 DLL 已经进入 Unity 进程之后，帮我们枚举 Unity 运行时中的程序集、类、字段和方法。这样就不需要自己手动去调用一堆 `mono_*` API，也不需要自己维护字段偏移。前面已经确认这个游戏加载的是 `mono-2.0-bdwgc.dll`，所以这里直接按 Mono 初始化。

这里先准备两个状态变量：`g_unityResolveInitialized` 表示 UnityResolve 是否初始化过，`g_setMentalHooksInitialized` 避免重复安装同一个 Hook。

```cpp dllmain.cpp
bool g_unityResolveInitialized = false;
bool g_setMentalHooksInitialized = false;
```

```cpp dllmain.cpp
void InitializeUnityResolve()
{
    if (g_unityResolveInitialized)
        return;

    while (!GetModuleHandleW(L"mono-2.0-bdwgc.dll"))
        Sleep(100);

    UnityResolve::Init(GetModuleHandleW(L"mono-2.0-bdwgc.dll"), UnityResolve::Mode::Mono);
    g_unityResolveInitialized = true;
}
```

`LockMental`这个功能是拦截游戏修改心理值的入口函数。前面在 dnSpy 里已经看到 `PlayerData.SetMental(float value)` 会对 `currentMental` 赋值，所以这里直接 Hook `PlayerData.SetMental(System.Single)`。

先准备函数指针和开关：

```cpp dllmain.cpp
using SetMentalFn = void(UNITY_CALLING_CONVENTION*)(void* instance, float value);

std::atomic_bool g_lockMental = false;
SetMentalFn g_originalPlayerDataSetMental = nullptr;
```

Hook 函数的逻辑很直接：如果菜单里的 `LockMental` 打开，就不再调用原函数；如果没打开，就把调用转发回原函数。

```cpp dllmain.cpp
void UNITY_CALLING_CONVENTION HookPlayerDataSetMental(void* instance, float value)
{
    if (g_lockMental.load())
        return;

    g_originalPlayerDataSetMental(instance, value);
}
```

这里的 `HookPlayerDataSetMental` 就是 MinHook 最终跳转进来的 detour 函数。它的函数签名必须和原来的 `PlayerData.SetMental(System.Single)` 对齐：`instance` 对应当前被调用的 `PlayerData` 实例，`value` 对应游戏准备传给 `SetMental(float value)` 的新心理值。

当 `g_lockMental.load()` 为 `true` 时，函数直接 `return`，也就是吞掉这次 `SetMental` 调用。原来的 `PlayerData.SetMental` 不会执行，因此 `this.currentMental = Mathf.Clamp(value, 0f, this.maxMental);` 也不会执行，心理值就不会被游戏逻辑继续改低。

当 `g_lockMental.load()` 为 `false` 时，函数会调用 `g_originalPlayerDataSetMental(instance, value)`。这个指针是 `MH_CreateHook` 填回来的原函数入口，用它把参数原样传回去，游戏就会按原本逻辑正常执行 `PlayerData.SetMental(System.Single)`。

再通过 UnityResolve 找到这个托管方法，并用 `Cast<void, void*, float>()` 拿到可以被 MinHook 使用的地址：

```cpp dllmain.cpp
void CreateManagedHook(void* target, LPVOID detour, LPVOID* original)
{
    MH_CreateHook(target, detour, original);
    MH_EnableHook(target);
}

bool InitializeSetMentalHooks()
{
    if (g_setMentalHooksInitialized)
        return true;

    InitializeUnityResolve();

    auto playerDataTarget = UnityResolve::Get("Assembly-CSharp.dll")
        ->Get("PlayerData")
        ->Get<UnityResolve::Method>("SetMental", { "System.Single" })
        ->Cast<void, void*, float>();

    CreateManagedHook(
        reinterpret_cast<void*>(playerDataTarget),
        &HookPlayerDataSetMental,
        reinterpret_cast<LPVOID*>(&g_originalPlayerDataSetMental));

    g_setMentalHooksInitialized = true;
    return true;
}
```

最后把 `InitializeSetMentalHooks()` 接到 ImGui 菜单里。菜单每帧都会尝试初始化一次，初始化成功后 `setMentalReady` 显示 `ready`；勾选 `LockMental` 后，`g_lockMental` 变为 `true`，上面的 Hook 函数就会直接拦截 `PlayerData.SetMental(System.Single)` 调用。

```cpp dllmain.cpp
const bool setMentalReady = InitializeSetMentalHooks();
bool lockMental = g_lockMental.load();
if (ImGui::Checkbox("LockMental", &lockMental))
    g_lockMental.store(lockMental);

ImGui::SameLine();
ImGui::TextDisabled(setMentalReady ? "ready" : "waiting");
```

这样做的效果是：游戏原本仍然可以正常运行自己的逻辑，但只要它试图通过 `PlayerData.SetMental(System.Single)` 修改心理值，就会被我们的 Hook 截断。关闭菜单里的 `LockMental` 后，调用会重新进入原函数，游戏逻辑恢复正常。

最后编译生成 DLL，把带转发的 `version.dll` 放到游戏 exe 同级目录。进入游戏后按 `Insert` 打开菜单，看到 `LockMental` 旁边显示 `ready` 后勾选它，再触发一次会降低 Mental 的行为。如果 Mental 不再下降，就说明 `PlayerData.SetMental(System.Single)` 已经被成功拦截。
