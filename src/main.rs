use std::ffi::{c_void, CString};
use std::fs::File;
use std::io::prelude::*;
use std::io::Error;
use std::iter::successors;
use std::{io, process, ptr};
use opencl3::device::{cl_context, CL_DEVICE_NAME, CL_DEVICE_TYPE_GPU, Device};
use opencl3::platform::{get_platforms, Platform};
use opencl3::{command_queue::CommandQueue};
use opencl3::context::Context;
use std::process::exit;
use std::rc::Rc;
use ocl::core::{ClPlatformIdPtr, DeviceId, get_device_info};
use ocl::enums::Status;
use ocl::ffi::{cl_uint, clEnqueueNDRangeKernel, clEnqueueReadBuffer, clEnqueueWriteBuffer, clFinish};
use opencl3::context::context::create_context;
use opencl3::device;
use opencl3::error_codes::ClError;
use opencl3::kernel::Kernel;
use opencl3::memory::Buffer;
use opencl3::program::Program;

use opencl3::program::CL_BUILD_SUCCESS;
use opencl3::types::CL_TRUE;
use rand::{random, Rng};

const MAX_NUM_DEVICES: usize = 16;
const MY_GPU_INDEX: usize = 0;

use std::sync::{Arc, Mutex};
use once_cell::sync::{Lazy, OnceCell};

static COMMAND_QUEUE: Lazy<Mutex<Option<(Arc<CommandQueue>, Context)>>> = Lazy::new(|| Mutex::new(None));

fn set_command_queue(queue: CommandQueue, context: Context) {
    let q_arc = Arc::new(queue);
    let mut global_queue = COMMAND_QUEUE.lock().unwrap();
    *global_queue = Some((q_arc, context));
}

fn get_command_queue() -> Option<(Arc<CommandQueue>, Context)> {
    let global_queue = COMMAND_QUEUE.lock().unwrap();
    global_queue.clone()
}




fn create_device() -> Result<Context, ClError> {
    let platforms = get_platforms()?;
    let platform = &platforms[0];

    let device_ids = platform.get_devices(CL_DEVICE_TYPE_GPU)?;
    if device_ids.is_empty() {
        eprintln!("No GPU device found.");
        process::exit(1);
    }

    let devices: Vec<Device> = device_ids.into_iter().map(Device::from).collect();

    for (i, device) in devices.iter().enumerate() {
        println!("GPU at index {}: {}", i, device.name().unwrap());
    }

    let selected_device = if MY_GPU_INDEX >= devices.len() {
        eprintln!("GPU at index {} not found, choosing the first one.", MY_GPU_INDEX);
        &devices[0]
    } else {
        &devices[MY_GPU_INDEX]
    };

    println!("Selected GPU Device: {}", selected_device.name().unwrap());

    let context = Context::from_device(selected_device)?;
    Ok(context)
}

unsafe fn create_kernel(source: &str) -> io::Result<Kernel> {
    println!("Loading kernel code..");
    let platform = opencl3::platform::get_platforms().unwrap()[0];
    let device_ids = platform.get_devices(CL_DEVICE_TYPE_GPU).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let devices: Vec<Device> = device_ids.into_iter().map(Device::from).collect();
    let device = &devices[0];
    let context = Context::from_device(&device).unwrap();

    let program = Program::create_and_build_from_source(
        &context,
        source,
        "",
    ).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    println!("Building kernel code..");
    if program.get_build_status(device.id()).unwrap() == CL_BUILD_SUCCESS {
        println!("Creating kernel...");
        let kernel = Kernel::create(&program, "sha256_crypt_kernel").unwrap();

        println!("Creating queue...");
        let command_queue = CommandQueue::create_with_properties(
            &context,
            context.default_device(),
            opencl3::command_queue::CL_QUEUE_PROFILING_ENABLE,
            0).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        Ok(kernel)
    } else {
        Err(io::Error::new(io::ErrorKind::Other, "Kernel build failed"))
    }
}


fn load_source() -> Result<String, Error> {
    let mut file = match File::open("./src/sha256_opencl.cl") {
        Err(_) => panic!("Could not load kernel, please place the sha256_opencl.cl in the same directory."),
        Ok(file) => file,
    };
    let mut source_str = String::new();
    match file.read_to_string(&mut source_str) {
        Err(_) => panic!("Error reading the file."),
        Ok(_) => return Ok(source_str),
    };
}

unsafe fn set_arg<T>(kernel: &Kernel, arg_index: usize, arg: &T) -> Result<(), ClError> {
    kernel.set_arg(arg_index as cl_uint, arg)

}

const RESULT_SIZE: usize = 4; // Replace with the actual size
const INPUT_STRIDE: usize = 128; // Replace with the actual size
const NPAR: usize = 32; // Replace with the actual size


unsafe fn create_clobj(context: &Context, kernel: Kernel)
                       -> Result<(opencl3::memory::Buffer<u8>, opencl3::memory::Buffer<u32>, opencl3::memory::Buffer<u8>, opencl3::memory::Buffer<u8>, opencl3::memory::Buffer<u32>), ClError>
{
    let pinned_saved_keys = opencl3::memory::Buffer::<u8>::create(
        context,
        opencl3::memory::CL_MEM_READ_WRITE | opencl3::memory::CL_MEM_ALLOC_HOST_PTR,
        INPUT_STRIDE,
        std::ptr::null_mut(),
    )?;

    let pinned_partial_hashes = opencl3::memory::Buffer::<u32>::create(
        context,
        opencl3::memory::CL_MEM_READ_WRITE | opencl3::memory::CL_MEM_ALLOC_HOST_PTR,
        (RESULT_SIZE * NPAR) * std::mem::size_of::<u32>(),
        std::ptr::null_mut(), // Replace None with std::ptr::null_mut()
    )?;

    let buffer_out = opencl3::memory::Buffer::<u8>::create(
        context,
        opencl3::memory::CL_MEM_WRITE_ONLY,
        (RESULT_SIZE * NPAR) * std::mem::size_of::<u32>(),
        std::ptr::null_mut(),
    )?;

    let buffer_keys = opencl3::memory::Buffer::<u8>::create(
        context,
        opencl3::memory::CL_MEM_READ_ONLY,
        INPUT_STRIDE,
        std::ptr::null_mut(),
    )?;

    let data_info = opencl3::memory::Buffer::<u32>::create(
        context,
        opencl3::memory::CL_MEM_READ_ONLY,
        3 * std::mem::size_of::<u32>(),
        std::ptr::null_mut(),
    )?;

    // Don't forget to set the kernel parameters
    kernel.set_arg(0 , &data_info)?;
    kernel.set_arg(1 , &buffer_keys)?;
    kernel.set_arg(2 , &buffer_out )?;

    Ok((pinned_saved_keys, pinned_partial_hashes, buffer_keys , buffer_out, data_info))
}

const SHA256_PLAINTEXT_LENGTH: i32 = 64;
const result_size: i32 = 16;
const npar: i32 = (1<<20);


fn sha256_init(){
    let source = match load_source() {
        Err(why) => panic!("couldn't read file: {}", why),
        Ok(source) => {
            println!("{}", source);
            source
        },
    };

    let context = create_device().expect("Couldn't create device.");
    let kernel = unsafe { create_kernel(&source).expect("Couldn't create kernel.") };
    let obj = unsafe { create_clobj(&context, kernel) };
}

fn main() {
    let hostname = "benchmark"; // Replace with actual host name
    let npar = 0; // Replace with actual npar value
    let nloops = 0; // Replace with actual nloops value
    let result = vec![0u8; 256];
    let bytes = vec![0u8; 67];

    sha256_init();

    if hostname == "benchmark" {
        let time_1 = std::time::Instant::now();
        let mut n_hashes = 0;

        while time_1.elapsed().as_secs_f64() < 1.1 {
            let result = &mut result.clone();
            let bytes = &bytes.clone();
            sha256_crypt(bytes.as_ptr(), 67, 2, 6, 1, result.as_mut_ptr()); // no idea how to implement this in rust
            n_hashes += npar * nloops;
        }

        let elapsed = time_1.elapsed().as_secs_f64();
        let hash_rate = (n_hashes as f64) / elapsed;
        println!("hash rate: {}/s", hash_rate);
        exit(0);
    }
}