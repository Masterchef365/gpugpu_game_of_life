mod hardware_query;
use anyhow::Result;
use erupt::{
    cstr,
    extensions::{ext_debug_utils, khr_surface, khr_swapchain},
    utils::{allocator::*, surface},
    vk1_0 as vk, DeviceLoader, EntryLoader, InstanceLoader,
};
use hardware_query::HardwareSelection;
use std::{ffi::CString, os::raw::c_char};
use winit::{event_loop::EventLoop, window::WindowBuilder};

fn main() -> Result<()> {
    // Windowing
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop)?;

    // App info
    let entry = EntryLoader::new()?;
    println!(
        "Vulkan Instance {}.{}.{}",
        vk::version_major(entry.instance_version()),
        vk::version_minor(entry.instance_version()),
        vk::version_patch(entry.instance_version())
    );

    let application_name = CString::new("GPUGPU GOL")?;
    let engine_name = CString::new("GPUGPU GOL")?;
    let app_info = vk::ApplicationInfoBuilder::new()
        .application_name(&application_name)
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    // Set up validation layers and features
    const LAYER_KHRONOS_VALIDATION: *const c_char = cstr!("VK_LAYER_KHRONOS_validation");

    let mut instance_extensions = surface::enumerate_required_extensions(&window).result()?;
    if cfg!(debug_assertions) {
        instance_extensions.push(ext_debug_utils::EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    let mut instance_layers = Vec::new();
    if cfg!(debug_assertions) {
        instance_layers.push(LAYER_KHRONOS_VALIDATION);
    }

    let device_extensions = vec![khr_swapchain::KHR_SWAPCHAIN_EXTENSION_NAME];

    let mut device_layers = Vec::new();
    if cfg!(debug_assertions) {
        device_layers.push(LAYER_KHRONOS_VALIDATION);
    }

    // Create instance
    let create_info = vk::InstanceCreateInfoBuilder::new()
        .application_info(&app_info)
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&instance_layers);

    let mut instance = InstanceLoader::new(&entry, &create_info, None)?;

    // Create surface
    let surface = unsafe { surface::create_surface(&mut instance, &window, None) }.result()?;

    // Find hardware
    let HardwareSelection {
        physical_device,
        queue_family,
        format,
        present_mode,
        ..
    } = HardwareSelection::query(&instance, surface, &device_extensions)?;

    // Set up device and queues
    let queue_create_info = vec![vk::DeviceQueueCreateInfoBuilder::new()
        .queue_family_index(queue_family)
        .queue_priorities(&[1.0])];
    let features = vk::PhysicalDeviceFeaturesBuilder::new();

    let create_info = vk::DeviceCreateInfoBuilder::new()
        .queue_create_infos(&queue_create_info)
        .enabled_features(&features)
        .enabled_extension_names(&device_extensions)
        .enabled_layer_names(&device_layers);

    let device = DeviceLoader::new(&instance, physical_device, &create_info, None)?;
    let queue = unsafe { device.get_device_queue(queue_family, 0, None) };

    // Create swapchain
    let surface_caps = unsafe {
        instance.get_physical_device_surface_capabilities_khr(physical_device, surface, None)
    }
    .result()?;
    let mut image_count = surface_caps.min_image_count + 1;
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }

    let create_info = khr_swapchain::SwapchainCreateInfoKHRBuilder::new()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(surface_caps.current_extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_caps.current_transform)
        .composite_alpha(khr_surface::CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(khr_swapchain::SwapchainKHR::null());

    let swapchain = unsafe { device.create_swapchain_khr(&create_info, None, None) }.result()?;
    let swapchain_images = unsafe { device.get_swapchain_images_khr(swapchain, None) }.result()?;

    // Create command pool
    let create_info = vk::CommandPoolCreateInfoBuilder::new()
        .queue_family_index(queue_family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let command_pool = unsafe { device.create_command_pool(&create_info, None, None) }.result()?;

    // Create command buffers
    let allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = unsafe { device.allocate_command_buffers(&allocate_info) }.result()?[0];

    // Create allocator
    let mut allocator =
        Allocator::new(&instance, physical_device, AllocatorCreateInfo::default()).result()?;

    let image_size_bytes =
        surface_caps.current_extent.width * surface_caps.current_extent.height * 4;

    // Create images
    let create_info = vk::BufferCreateInfoBuilder::new()
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .size(image_size_bytes as u64);

    let display_image = unsafe { device.create_buffer(&create_info, None, None) }.result()?;
    let display_image_mem = allocator
        .allocate(&device, display_image, MemoryTypeFinder::dynamic())
        .result()?;

    // Create synchronization primitives
    let create_info = vk::SemaphoreCreateInfoBuilder::new();

    // Whether or not the frame at this index is available (gpu-only)
    let image_available = 
        unsafe { device.create_semaphore(&create_info, None, None) }.result()?;

    // Whether or not the frame at this index is finished rendering (gpu-only)
    let render_finished =
        unsafe { device.create_semaphore(&create_info, None, None) }.result()?;

    // Main loop
    let mut time = 0;
    loop {
        // Get the index of the next swapchain image,
        // and set up a semaphore to be notified when it is ready.
        let image_index = unsafe {
            device.acquire_next_image_khr(swapchain, u64::MAX, Some(image_available), None, None)
        }
        .result()?;

        // Create image data
        let the_image = (0..image_size_bytes)
            .map(|v| {
                if (v as f32 + time as f32).cos() < 0.0 {
                    255
                } else {
                    0
                }
            })
        .collect::<Vec<_>>();

        // Map image data
        let mut map = display_image_mem.map(&device, ..).result()?;
        map.import(&the_image);
        map.unmap(&device).result()?;

        let swapchain_image = swapchain_images[image_index as usize];

        // Build command buffer
        let command_buffer = command_buffer;
        unsafe {
            let begin_info = vk::CommandBufferBeginInfoBuilder::new();
            device.reset_command_buffer(command_buffer, None).result()?;
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .result()?;

            // Transition display image from GENERAL to SRC_OPTIMAL, preserving contents
            // Transition swapchain image from SRC_KHR to DST_OPTIMAL
            let sub = vk::ImageSubresourceRangeBuilder::new()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build();

            let swapchain_image_barrier = vk::ImageMemoryBarrierBuilder::new()
                .image(swapchain_image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .subresource_range(sub);

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                None,
                &[],
                &[],
                &[swapchain_image_barrier],
            );

            // Copy from staging image to swapchain image
            let sub = vk::ImageSubresourceLayersBuilder::new()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1)
                .build();
            let off = vk::Offset3DBuilder::new().x(0).y(0).z(0).build();
            let extent = vk::Extent3DBuilder::new()
                .width(surface_caps.current_extent.width)
                .height(surface_caps.current_extent.height)
                .depth(1)
                .build();
            let copy_info = vk::BufferImageCopyBuilder::new()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(sub)
                .image_offset(off)
                .image_extent(extent);
            device.cmd_copy_buffer_to_image(
                command_buffer,
                display_image,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_info],
            );

            // Transition display image from SRC_OPTIMAL to GENERAL, preserving contents
            // Transition swapchain image from DST_OPTIMAL to SRC_KHR
            let sub = vk::ImageSubresourceRangeBuilder::new()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build();

            let swapchain_image_barrier = vk::ImageMemoryBarrierBuilder::new()
                .image(swapchain_image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .subresource_range(sub);

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                None,
                &[],
                &[],
                &[swapchain_image_barrier],
            );

            device.end_command_buffer(command_buffer).result()?;
        }

        // Submit command buffer to the queue, waiting on the image from the swapchain and
        // signalling "render finished" inside the GPU when done.
        let wait_semaphores = [image_available];
        let command_buffers = [command_buffer];
        let signal_semaphores = [render_finished];
        let submit_info = vk::SubmitInfoBuilder::new()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COMPUTE_SHADER])
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        unsafe {
            device
                .queue_submit(queue, &[submit_info], None)
                .unwrap()
        }

        let swapchains = [swapchain];
        let image_indices = [image_index];
        let present_info = khr_swapchain::PresentInfoKHRBuilder::new()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe { device.queue_present_khr(queue, &present_info) }.unwrap();

        unsafe {
            device.queue_wait_idle(queue).result()?;
        }

        time += 1;
    }
}

