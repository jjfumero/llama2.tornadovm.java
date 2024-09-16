/*
 * MIT License
 *
 * Copyright (c) 2024, Juan Fumero.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package io.github.mikepapadim.gpu.shared;

import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroBufferInteger;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroCommandList;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroCommandQueue;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroContext;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroDevice;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroDriver;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroKernel;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroModule;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.Pointer;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.Sizeof;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeAPIVersion;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeBuildLogHandle;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeCacheConfigFlag;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeCommandListDescriptor;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeCommandListHandle;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeCommandQueueDescriptor;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeCommandQueueGroupProperties;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeCommandQueueGroupPropertyFlags;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeCommandQueueHandle;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeCommandQueueMode;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeContextDescriptor;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeDeviceMemAllocDescriptor;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeDevicesHandle;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeDriverHandle;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeDriverProperties;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeGroupDispatch;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeHostMemAllocDescriptor;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeHostMemAllocFlags;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeInitFlag;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeKernelDescriptor;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeKernelHandle;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeModuleDescriptor;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeModuleFormat;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeModuleHandle;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeRelaxedAllocationLimitsExpDescriptor;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeRelaxedAllocationLimitsFlags;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeResult;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.Ze_Structure_Type;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.utils.LevelZeroUtils;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * ComputeBundle taken from {@url https://github.com/jjfumero/corellama2.levelzero}
 */
public class ComputeBundle {

    private LevelZeroDriver driver;
    private LevelZeroDevice device;
    private ZeDriverHandle driverHandler;
    private LevelZeroContext context;
    private ZeDevicesHandle deviceHandler;

    private LevelZeroCommandQueue commandQueue;
    private LevelZeroCommandList commandList;

    private final boolean INFO = true;

    private LevelZeroModule levelZeroModule;
    private ZeModuleHandle module;
    private ZeGroupDispatch dispatcherMatMul;

    private void initDriver() {
        // Create the Level Zero Driver
        driver = new LevelZeroDriver();
        int result = driver.zeInit(ZeInitFlag.ZE_INIT_FLAG_GPU_ONLY);
        LevelZeroUtils.errorLog("zeInit", result);

        int[] numDrivers = new int[1];
        result = driver.zeDriverGet(numDrivers, null);
        LevelZeroUtils.errorLog("zeDriverGet", result);

        driverHandler = new ZeDriverHandle(numDrivers[0]);
        result = driver.zeDriverGet(numDrivers, driverHandler);
        LevelZeroUtils.errorLog("zeDriverGet", result);

        result = driver.zeDriverGet(numDrivers, driverHandler);
        LevelZeroUtils.errorLog("zeDriverGet", result);
    }

    private void createContext() {
        // ============================================
        // Create the Context
        // ============================================
        // Create context Description
        ZeContextDescriptor contextDescription = new ZeContextDescriptor();
        // Create context object
        context = new LevelZeroContext(driverHandler, contextDescription);
        // Call native method for creating the context
        int result = context.zeContextCreate(driverHandler.getZe_driver_handle_t_ptr()[0]);
        LevelZeroUtils.errorLog("zeContextCreate", result);
    }

    private void getDeviceWithContext() {
        // Get number of devices in a driver
        int[] deviceCount = new int[1];
        int result = driver.zeDeviceGet(driverHandler, 0, deviceCount, null);
        LevelZeroUtils.errorLog("zeDeviceGet", result);

        // Instantiate a device Handler
        deviceHandler = new ZeDevicesHandle(deviceCount[0]);
        result = driver.zeDeviceGet(driverHandler, 0, deviceCount, deviceHandler);
        LevelZeroUtils.errorLog("zeDeviceGet", result);

        // Obtain device from the list
        device = driver.getDevice(driverHandler, 0);
    }

    private void printDeviceProperties() {
        // ============================================
        // Query driver properties
        // ============================================
        ZeDriverProperties driverProperties = new ZeDriverProperties(Ze_Structure_Type.ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES);
        int result = driver.zeDriverGetProperties(driverHandler, 0, driverProperties);
        LevelZeroUtils.errorLog("zeDriverGetProperties", result);

        System.out.println("Driver Version: " + driverProperties.getDriverVersion());

        ZeAPIVersion apiVersion = new ZeAPIVersion();
        result = driver.zeDriverGetApiVersion(driverHandler, 0, apiVersion);
        LevelZeroUtils.errorLog("zeDriverGetApiVersion", result);

        System.out.println("Level Zero API Version: " + apiVersion);
    }

    private void createCommandQueue() {
        // ============================================
        // Create a command queue
        // ============================================
        // A) Get the number of command queue groups
        int[] numQueueGroups = new int[1];
        int result = device.zeDeviceGetCommandQueueGroupProperties(device.getDeviceHandlerPtr(), numQueueGroups, null);
        LevelZeroUtils.errorLog("zeDeviceGetCommandQueueGroupProperties", result);

        if (numQueueGroups[0] == 0) {
            throw new RuntimeException("Number of Queue Groups is 0 for device: " + device.getDeviceProperties().getName());
        }

        ZeCommandQueueGroupProperties[] commandQueueGroupProperties = new ZeCommandQueueGroupProperties[numQueueGroups[0]];
        result = device.zeDeviceGetCommandQueueGroupProperties(device.getDeviceHandlerPtr(), numQueueGroups, commandQueueGroupProperties);
        LevelZeroUtils.errorLog("zeDeviceGetCommandQueueGroupProperties", result);
        for (ZeCommandQueueGroupProperties p : commandQueueGroupProperties) {
            System.out.println(p);
        }

        ZeCommandQueueHandle commandQueueHandle = new ZeCommandQueueHandle();
        commandQueue = new LevelZeroCommandQueue(context, commandQueueHandle);
        ZeCommandQueueDescriptor commandQueueDescription = new ZeCommandQueueDescriptor();

        for (int i = 0; i < numQueueGroups[0]; i++) {
            if ((commandQueueGroupProperties[i].getFlags()
                    & ZeCommandQueueGroupPropertyFlags.ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) == ZeCommandQueueGroupPropertyFlags.ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
                commandQueueDescription.setOrdinal(i);
            }
        }

        // B) Create the command queue via the context
        commandQueueDescription.setIndex(0);
        commandQueueDescription.setMode(ZeCommandQueueMode.ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS);
        // zeCommandQueueCreate(context, device, &cmdQueueDesc, &cmdQueue);
        result = context.zeCommandQueueCreate(context.getContextHandle().getContextPtr()[0], device.getDeviceHandlerPtr(), commandQueueDescription, commandQueueHandle);
        LevelZeroUtils.errorLog("zeCommandQueueCreate", result);

        // ============================================
        // Create a command list
        // ============================================
        ZeCommandListHandle zeCommandListHandler = new ZeCommandListHandle();
        commandList = new LevelZeroCommandList(context, zeCommandListHandler);
        ZeCommandListDescriptor commandListDescription = new ZeCommandListDescriptor();
        commandListDescription.setCommandQueueGroupOrdinal(commandQueueDescription.getOrdinal());
        result = context.zeCommandListCreate(context.getContextHandle().getContextPtr()[0], device.getDeviceHandlerPtr(), commandListDescription, zeCommandListHandler);
        LevelZeroUtils.errorLog("zeCommandListCreate", result);
    }

    private void compileSPIRVKernel(String spirvFile) {
        module = new ZeModuleHandle();
        ZeModuleDescriptor moduleDesc = new ZeModuleDescriptor();
        ZeBuildLogHandle buildLog = new ZeBuildLogHandle();
        moduleDesc.setFormat(ZeModuleFormat.ZE_MODULE_FORMAT_IL_SPIRV);
        moduleDesc.setBuildFlags("-ze-opt-level 2");

        int result = context.zeModuleCreate(context.getDefaultContextPtr(), device.getDeviceHandlerPtr(), moduleDesc, module, buildLog, spirvFile);
        LevelZeroUtils.errorLog("zeModuleCreate", result);

        if (result != ZeResult.ZE_RESULT_SUCCESS) {
            // Print Logs
            int[] sizeLog = new int[1];
            String[] errorMessage = new String[1];
            result = context.zeModuleBuildLogGetString(buildLog, sizeLog, errorMessage);
            System.out.println("LOGS::: " + sizeLog[0] + "  -- " + errorMessage[0]);
            LevelZeroUtils.errorLog("zeModuleBuildLogGetString", result);
            throw new RuntimeException("[Error] zeModuleCreate failed");
        }

        levelZeroModule = new LevelZeroModule(module, moduleDesc, buildLog);

        result = levelZeroModule.zeModuleBuildLogDestroy(buildLog);
        LevelZeroUtils.errorLog("zeModuleBuildLogDestroy", result);
    }

    public void initializeLevelZeroPlatform(String spirvFile) {
        initDriver();
        createContext();
        getDeviceWithContext();
        if (INFO) {
            printDeviceProperties();
        }
        createCommandQueue();
        compileSPIRVKernel(spirvFile);
    }

    public LevelZeroKernel createKernel(String kernelName) {
        ZeKernelDescriptor kernelDesc = new ZeKernelDescriptor();
        ZeKernelHandle kernel = new ZeKernelHandle();
        kernelDesc.setKernelName(kernelName);
        int result = levelZeroModule.zeKernelCreate(module.getPtrZeModuleHandle(), kernelDesc, kernel);
        LevelZeroUtils.errorLog("zeKernelCreate", result);

        LevelZeroKernel levelZeroKernel = new LevelZeroKernel(kernelDesc, kernel, levelZeroModule);

        result = levelZeroKernel.zeKernelSetCacheConfig(kernel.getPtrZeKernelHandle(), ZeCacheConfigFlag.ZE_CACHE_CONFIG_FLAG_LARGE_SLM);
        LevelZeroUtils.errorLog("zeKernelSetCacheConfig", result);
        return levelZeroKernel;
    }

    public MemObject allocateSharedWithSegment(long numElements) {
        final long bufferSize = numElements * Sizeof.FLOAT.getNumBytes();
        LevelZeroBufferInteger buffer = new LevelZeroBufferInteger();
        ZeDeviceMemAllocDescriptor deviceMemAllocDesc = new ZeDeviceMemAllocDescriptor();
        ZeHostMemAllocDescriptor hostMemAllocDesc = new ZeHostMemAllocDescriptor();

        hostMemAllocDesc.setFlags(ZeHostMemAllocFlags.ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED);

        // Extended memory support for large buffers
        ZeRelaxedAllocationLimitsExpDescriptor relaxedAllocationLimitsExpDescriptor = new ZeRelaxedAllocationLimitsExpDescriptor();
        relaxedAllocationLimitsExpDescriptor.setFlags(ZeRelaxedAllocationLimitsFlags.ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE);
        relaxedAllocationLimitsExpDescriptor.materialize();
        deviceMemAllocDesc.setNext(relaxedAllocationLimitsExpDescriptor);

        int result = context.zeMemAllocShared(context.getContextHandle().getContextPtr()[0], deviceMemAllocDesc, hostMemAllocDesc, bufferSize, 1, device.getDeviceHandlerPtr(), buffer);
        LevelZeroUtils.errorLog("zeMemAllocShared", result);

        // Allocate Panama Region using the Level Zero Buffer Pointer
        MemorySegment segment = MemorySegment.ofAddress(buffer.getPtrBuffer()).reinterpret(bufferSize);

        return new MemObject(segment, buffer);
    }

    private void launchAndSync(LevelZeroKernel levelZeroKernel, ZeGroupDispatch dispatch) {
        ZeKernelHandle kernel = levelZeroKernel.getKernelHandle();
        // Launch the kernel on the Intel Integrated GPU
        int result = commandList.zeCommandListAppendLaunchKernel(commandList.getCommandListHandlerPtr(), kernel.getPtrZeKernelHandle(), dispatch, null, 0, null);
        LevelZeroUtils.errorLog("zeCommandListAppendLaunchKernel", result);

        result = commandList.zeCommandListClose(commandList.getCommandListHandlerPtr());
        LevelZeroUtils.errorLog("zeCommandListClose", result);

        result = commandQueue.zeCommandQueueExecuteCommandLists(commandQueue.getCommandQueueHandlerPtr(),  1, commandList.getCommandListHandler(), null);
        LevelZeroUtils.errorLog("zeCommandQueueExecuteCommandLists", result);

        result = commandQueue.zeCommandQueueSynchronize(commandQueue.getCommandQueueHandlerPtr(), Long.MAX_VALUE);
        LevelZeroUtils.errorLog("zeCommandQueueSynchronize", result);

        result = commandList.zeCommandListReset(commandList.getCommandListHandlerPtr());
        LevelZeroUtils.errorLog("zeCommandListReset", result);
    }

    public void runKernelTesting(LevelZeroKernel levelZeroKernel, int numElements, LevelZeroBufferInteger bufferA, LevelZeroBufferInteger bufferB) {

        int[] groupSizeX = new int[] { numElements };
        int[] groupSizeY = new int[] { 1 };
        int[] groupSizeZ = new int[] { 1 };

        ZeKernelHandle kernel = levelZeroKernel.getKernelHandle();

        int result = levelZeroKernel.zeKernelSuggestGroupSize(kernel.getPtrZeKernelHandle(), numElements, 1, 1, groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSuggestGroupSize", result);

        result = levelZeroKernel.zeKernelSetGroupSize(kernel.getPtrZeKernelHandle(), groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSetGroupSize", result);

        result = levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 0, Sizeof.POINTER.getNumBytes(), bufferA);
        result |= levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 1, Sizeof.POINTER.getNumBytes(), bufferB);
        LevelZeroUtils.errorLog("zeKernelSetArgumentValue", result);

        ZeGroupDispatch dispatch = new ZeGroupDispatch();
        dispatch.setGroupCountX(numElements / groupSizeX[0]);
        dispatch.setGroupCountY(1);
        dispatch.setGroupCountZ(1);

        launchAndSync(levelZeroKernel, dispatch);
    }

    public void runRMSNorm1(LevelZeroKernel levelZeroKernel, int numElements, final int groupSize, LevelZeroBufferInteger dOutput, LevelZeroBufferInteger dX) {

        int[] groupSizeX = new int[] { numElements };
        int[] groupSizeY = new int[] { 1 };
        int[] groupSizeZ = new int[] { 1 };

        ZeKernelHandle kernel = levelZeroKernel.getKernelHandle();

        int result = levelZeroKernel.zeKernelSuggestGroupSize(kernel.getPtrZeKernelHandle(), numElements, 1, 1, groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSuggestGroupSize", result);

        result = levelZeroKernel.zeKernelSetGroupSize(kernel.getPtrZeKernelHandle(), groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSetGroupSize", result);

        result = levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 0, Sizeof.POINTER.getNumBytes(), dOutput);
        result |= levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 1, Sizeof.POINTER.getNumBytes(), dX);
        int sharedMemorySize = groupSize * Sizeof.FLOAT.getNumBytes();
        result |= levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 2, sharedMemorySize, null);
        LevelZeroUtils.errorLog("zeKernelSetArgumentValue", result);

        ZeGroupDispatch dispatch = new ZeGroupDispatch();
        dispatch.setGroupCountX(numElements / groupSizeX[0]);
        dispatch.setGroupCountY(1);
        dispatch.setGroupCountZ(1);

        launchAndSync(levelZeroKernel, dispatch);
    }

    public void runRMSNorm2(LevelZeroKernel levelZeroKernel, int numElements, final float ss, LevelZeroBufferInteger... parameters) {

        int[] groupSizeX = new int[] { numElements };
        int[] groupSizeY = new int[] { 1 };
        int[] groupSizeZ = new int[] { 1 };

        ZeKernelHandle kernel = levelZeroKernel.getKernelHandle();

        int result = levelZeroKernel.zeKernelSuggestGroupSize(kernel.getPtrZeKernelHandle(), numElements, 1, 1, groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSuggestGroupSize", result);

        result = levelZeroKernel.zeKernelSetGroupSize(kernel.getPtrZeKernelHandle(), groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSetGroupSize", result);

        for (int i = 0; i < parameters.length; i++) {
            result |= levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), i, Sizeof.POINTER.getNumBytes(), parameters[i]);
        }
        result |= levelZeroKernel.zeKernelSetArgumentValuePrimitive(kernel.getPtrZeKernelHandle(), parameters.length, Sizeof.FLOAT.getNumBytes(), Pointer.to(ss));
        LevelZeroUtils.errorLog("zeKernelSetArgumentValue", result);

        ZeGroupDispatch dispatch = new ZeGroupDispatch();
        dispatch.setGroupCountX(numElements / groupSizeX[0]);
        dispatch.setGroupCountY(1);
        dispatch.setGroupCountZ(1);

        launchAndSync(levelZeroKernel, dispatch);
    }

    public void runSoftMax1(LevelZeroKernel levelZeroKernel, int numElements, final int groupSize, LevelZeroBufferInteger dOutput, LevelZeroBufferInteger dX) {
        int[] groupSizeX = new int[] { numElements };
        int[] groupSizeY = new int[] { 1 };
        int[] groupSizeZ = new int[] { 1 };

        ZeKernelHandle kernel = levelZeroKernel.getKernelHandle();

        int result = levelZeroKernel.zeKernelSuggestGroupSize(kernel.getPtrZeKernelHandle(), numElements, 1, 1, groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSuggestGroupSize", result);

        result = levelZeroKernel.zeKernelSetGroupSize(kernel.getPtrZeKernelHandle(), groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSetGroupSize", result);

        result = levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 0, Sizeof.POINTER.getNumBytes(), dOutput);
        result |= levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 1, Sizeof.POINTER.getNumBytes(), dX);
        int sharedMemorySize = groupSize * Sizeof.FLOAT.getNumBytes();
        result |= levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 2, sharedMemorySize, null);
        LevelZeroUtils.errorLog("zeKernelSetArgumentValue", result);

        ZeGroupDispatch dispatch = new ZeGroupDispatch();
        dispatch.setGroupCountX(numElements / groupSizeX[0]);
        dispatch.setGroupCountY(1);
        dispatch.setGroupCountZ(1);

        launchAndSync(levelZeroKernel, dispatch);
    }

    public void runSoftMax2(LevelZeroKernel levelZeroKernel, int numElements, final int groupSize, LevelZeroBufferInteger dOutput, LevelZeroBufferInteger dX, float maxValue) {
        int[] groupSizeX = new int[] { numElements };
        int[] groupSizeY = new int[] { 1 };
        int[] groupSizeZ = new int[] { 1 };

        ZeKernelHandle kernel = levelZeroKernel.getKernelHandle();

        int result = levelZeroKernel.zeKernelSuggestGroupSize(kernel.getPtrZeKernelHandle(), numElements, 1, 1, groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSuggestGroupSize", result);

        result = levelZeroKernel.zeKernelSetGroupSize(kernel.getPtrZeKernelHandle(), groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSetGroupSize", result);

        result = levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 0, Sizeof.POINTER.getNumBytes(), dOutput);
        LevelZeroUtils.errorLog("zeKernelSetArgumentValue 0", result);
        result = levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 1, Sizeof.POINTER.getNumBytes(), dX);
        LevelZeroUtils.errorLog("zeKernelSetArgumentValue 1", result);
        final int sharedMemorySize = groupSize * Sizeof.INT.getNumBytes();
        result = levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 2, sharedMemorySize, null);
        LevelZeroUtils.errorLog("zeKernelSetArgumentValue 2", result);
        result = levelZeroKernel.zeKernelSetArgumentValuePrimitive(kernel.getPtrZeKernelHandle(), 3, Sizeof.FLOAT.getNumBytes(), Pointer.to(maxValue));
        LevelZeroUtils.errorLog("zeKernelSetArgumentValue 3", result);

        ZeGroupDispatch dispatch = new ZeGroupDispatch();
        dispatch.setGroupCountX(numElements / groupSizeX[0]);
        dispatch.setGroupCountY(1);
        dispatch.setGroupCountZ(1);

        launchAndSync(levelZeroKernel, dispatch);
    }

    public void runSoftMax3(LevelZeroKernel levelZeroKernel, int numElements, LevelZeroBufferInteger dX, float valSum) {
        int[] groupSizeX = new int[] { numElements };
        int[] groupSizeY = new int[] { 1 };
        int[] groupSizeZ = new int[] { 1 };

        ZeKernelHandle kernel = levelZeroKernel.getKernelHandle();

        // Compute block of threads
        int result = levelZeroKernel.zeKernelSuggestGroupSize(kernel.getPtrZeKernelHandle(), numElements, 1, 1, groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSuggestGroupSize", result);
        result = levelZeroKernel.zeKernelSetGroupSize(kernel.getPtrZeKernelHandle(), groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSetGroupSize", result);

        // Set Parameters
        result = levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 0, Sizeof.POINTER.getNumBytes(), dX);
        LevelZeroUtils.errorLog("zeKernelSetArgumentValue 0", result);
        result = levelZeroKernel.zeKernelSetArgumentValuePrimitive(kernel.getPtrZeKernelHandle(), 1, Sizeof.FLOAT.getNumBytes(), Pointer.to(valSum));
        LevelZeroUtils.errorLog("zeKernelSetArgumentValue 1", result);

        // Dispatch
        ZeGroupDispatch dispatch = new ZeGroupDispatch();
        dispatch.setGroupCountX(numElements / groupSizeX[0]);
        dispatch.setGroupCountY(1);
        dispatch.setGroupCountZ(1);

        launchAndSync(levelZeroKernel, dispatch);
    }

    public void dispatchMatMul(LevelZeroKernel levelZeroKernel, ZeGroupDispatch dispatch) {
        launchAndSync(levelZeroKernel, dispatch);
    }

    public ZeGroupDispatch runMatMul(LevelZeroKernel levelZeroKernel, LevelZeroBufferInteger dXout, LevelZeroBufferInteger dX, LevelZeroBufferInteger dW, int numElements, int numThreads) {

        int[] groupSizeX = new int[] { numThreads };
        int[] groupSizeY = new int[] { 1 };
        int[] groupSizeZ = new int[] { 1 };

        ZeKernelHandle kernel = levelZeroKernel.getKernelHandle();

        // Compute block of threads
        int result = levelZeroKernel.zeKernelSuggestGroupSize(kernel.getPtrZeKernelHandle(), numThreads, 1, 1, groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSuggestGroupSize", result);
        result = levelZeroKernel.zeKernelSetGroupSize(kernel.getPtrZeKernelHandle(), groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSetGroupSize", result);

        // Set Parameters
        result = levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 0, Sizeof.POINTER.getNumBytes(), dXout);
        result |= levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 1, Sizeof.POINTER.getNumBytes(), dX);
        result |= levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 2, Sizeof.POINTER.getNumBytes(), dW);
        result |= levelZeroKernel.zeKernelSetArgumentValuePrimitive(kernel.getPtrZeKernelHandle(), 3, Sizeof.INT.getNumBytes(), Pointer.to(numElements));
        LevelZeroUtils.errorLog("zeKernelSetArgumentValue: ", result);
        
        // Dispatch
        ZeGroupDispatch dispatch = new ZeGroupDispatch();
        dispatch.setGroupCountX(numThreads / groupSizeX[0]);
        dispatch.setGroupCountY(1);
        dispatch.setGroupCountZ(1);

        launchAndSync(levelZeroKernel, dispatch);
        return dispatch;
    }

    public void print(MemorySegment segment) {
        // Print Some Elements
        IntStream.range(0, 10).forEach(i -> {
            float atIndex = segment.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            System.out.println(atIndex);
        });
    }

    public void testingInitData(MemorySegment segment, int size) {
        // Initialize Data
        for (int i = 0; i < size; i++) {
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i,  100 + i);
        }
    }

    public void init1DRandom(MemorySegment segment, int size) {
        // Initialize Data
        Random r = new Random(71);
        for (int i = 0; i < size; i++) {
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, r.nextFloat());
        }
    }

    public void init2DRandom(MemorySegment segment, int size) {
        // Initialize Data
        Random r = new Random(71);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                segment.setAtIndex(ValueLayout.JAVA_FLOAT, i * size + j, r.nextFloat());
            }
        }
    }

    public boolean testingCheckResult(MemorySegment outputSegment, int size) {
        for (int i = 0; i < size; i++) {
            var val = outputSegment.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (val != (100 + i)) {
                return false;
            }
        }
        return true;
    }

    public void setDispatchMatMul(ZeGroupDispatch dispatcher) {
        this.dispatcherMatMul = dispatcher;
    }

    public ZeGroupDispatch getMatMulDispatcher() {
        return this.dispatcherMatMul;
    }

    public boolean isThreadConfigurationDone() {
        return this.dispatcherMatMul != null;
    }
}
