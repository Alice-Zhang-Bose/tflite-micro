## Environment Setup
1. In addition to the cygwin packages needed for walnut [listed here](https://wiki.bose.com/pages/viewpage.action?spaceKey=CER&title=Walnut+-+Development+Environment+Setup#WalnutDevelopmentEnvironmentSetup-TerminalSetup:Cygwin), install `wget` and `unzip` packages in cygwin.
2. In `git bash``, clone this repo and checkout this specific commit: `8c0170a5bc64b173e8bdda3e8e90dffe2a892a98`.
   ```
   $ git clone git@github.com:Alice-Zhang-Bose/tflite-micro.git
   $ cd tflite-micro
   $ git checkout 8c0170a5bc64b173e8bdda3e8e90dffe2a892a98
   $ git cherry-pick 625c3a467db4a949edbdafeaa77333a2c6179b35
   ```
3. In `cygwin`, run `dos2unix` on all files in this repo.
    ```
    $ find . -type f -print0 | xargs -0 dos2unix
    ```
4. Create python virtual environment in the root directory of this repo and install required packages according to requirements.txt file.
    ```
    $ python -m venv tflite_env
    $ source tflite_env/Scripts/activate
    $ pip install tensorflow-io-gcs-filesystem==0.31.0
    $ pip install markdown==3.4.3
    $ pip install tensorflow-intel==2.12.0
    $ pip install -r third_party/python_requirements.txt
    ```
5. Some (~3) packages fail the install from the `requirements.txt` file. In those cases, I removed the hash from the `requirements.txt` and tried running `pip install -r third_party/python_requirements.txt` again or manually installed those packages with `pip install <package>`.
6. Export environment variables in `.tflmrc`. NOTE: this file currently assumes the root directory of this repo lives at `/c/git/`. Edit `source /cygdrive/c/git/tflite-micro/tflite_env/Scripts/activate` in the script to match the path of your virtual environment.
   ```
   source .tflmrc
   ```
7. The `hello_world` test should now pass.
   ```
   make -f  tensorflow/lite/micro/tools/make/Makefile test_hello_world_test
   ```

## Build and test keyword detection (micro_speech) example
1. Build the `micro_speech` example with the Xtensa tools for the adau1797 core in cygwin.  
    ```
    $ make -f  tensorflow/lite/micro/tools/make/Makefile micro_speech TARGET=xtensa TARGET_ARCH=hifi3
    ```
    If successful, this command should generate a bin file under `gen/xtensa_hifi3_default/bin/micro_speech`.
2. Run the `micro_speech` test.  
    ```
    $ make -f  tensorflow/lite/micro/tools/make/Makefile test_micro_speech_test TARGET=xtensa TARGET_ARCH=hifi3
    ```
    If successful, this command should print:
    ```
    tensorflow/lite/micro/tools/make/test_latency_log.sh micro_speech_test tensorflow/lite/micro/testing/test_xtensa_binary.sh gen/xtensa_hifi3_default/bin/micro_speech_test '~~~ALL TESTS PASSED~~~' xtensa
    Testing TestInvoke
    Ran successfully

    1/1 tests passed
    ~~~ALL TESTS PASSED~~~

    Running micro_speech_test took 60.999 seconds
    ```

## Deploy program to board 
TODO: Verify that this actually loads something meaningful onto the board
1. Load the program onto the board.  
    a. In one cygwin terminal, launch Xtensa OCD:
    ```
    $ "C:/Program Files (x86)/Tensilica/Xtensa OCD Daemon 14.08/xt-ocd" --config=C:/Users/az1058168/Desktop/topology_no_security.xml -dTD=30 -T 20
    ```
    Note: Check that the topology file has the USB serial number that matches the serial number written on the physical Jlink device.  
    b. In another cygwin terminal, launch Xtensa GDB:
    ```
    $ "C:/usr/xtensa/XtDevTools/install/tools/RI-2021.8-win32/XtensaTools/bin/xt-gdb.exe" --xtensa-system=C:/usr/xtensa/XtDevTools/install/builds/RI-2021.8-win32/adau1797_2021_8/config --xtensa-core=adau1797_2021_8 gen/xtensa_hifi3_default/bin/micro_speech
    ```
2. After Xtensa GDB is launched, in the Xtensa GDB terminal run the following commands sequentially:  
    a. `target remote localhost:20000`  
    b. `reset`  
    c. `load`  
    After running load, the debugger should print something like the following: 
    ```
    (xt-gdb) load
    Loading section .MemoryExceptionVector.literal, size 0x4 lma 0x40000
    Loading section .UserExceptionVector.literal, size 0x4 lma 0x40004
    Loading section .ResetVector.text, size 0x188 lma 0x60000
    Loading section .MemoryExceptionVector.text, size 0x50 lma 0x60310
    Loading section .WindowVectors.text, size 0x16c lma 0x60800
    Loading section .Level2InterruptVector.text, size 0x8 lma 0x6097c
    Loading section .Level3InterruptVector.text, size 0x8 lma 0x6099c
    Loading section .Level4InterruptVector.text, size 0x8 lma 0x609bc
    Loading section .Level5InterruptVector.text, size 0x8 lma 0x609dc
    Loading section .DebugExceptionVector.text, size 0x8 lma 0x609fc
    Loading section .NMIExceptionVector.text, size 0x4 lma 0x60a1c
    Loading section .KernelExceptionVector.text, size 0x10 lma 0x60a3c
    Loading section .UserExceptionVector.text, size 0xc lma 0x60a5c
    Loading section .DoubleExceptionVector.text, size 0x8 lma 0x60a7c
    Loading section .clib.rodata, size 0x120 lma 0x10000000
    Loading section .rodata, size 0x8d88 lma 0x10000120
    Loading section .text, size 0x15c28 lma 0x10008ea8
    Loading section .clib.data, size 0x134 lma 0x1001ead0
    Loading section .rtos.percpu.data, size 0x300 lma 0x1001ec08
    Loading section .data, size 0x48 lma 0x1001ef10
    Start address 0x00060000, load size 127720
    Transfer rate: 86 KB/sec, 4730 bytes/write.
    ```
    To end the debugging session, run:  
    d. `detach`  
    e. `quit`  

