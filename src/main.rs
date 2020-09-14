mod hardware_query;
use anyhow::{Context, Result};
use erupt::{
    cstr,
    extensions::{ext_debug_utils, khr_surface, khr_swapchain},
    utils::{allocator::*, decode_spv, surface},
    vk1_0 as vk, DeviceLoader, EntryLoader, InstanceLoader,
};
use hardware_query::HardwareSelection;
use std::{ffi::CString, os::raw::c_char};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
    dpi::{PhysicalPosition, PhysicalSize, Size},
};

const COLOR_FORMAT_SWAP: vk::Format = vk::Format::B8G8R8A8_SRGB;
const COLOR_FORMAT_INTERMEDIATE: vk::Format = vk::Format::R8G8B8A8_UINT;
const DATA_FORMAT: vk::Format = vk::Format::R8_UINT;

fn main() -> Result<()> {
    // Windowing
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(Size::Physical(PhysicalSize::new(1000, 1000)))
        .with_resizable(false)
        .build(&event_loop)?;

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
        .image_format(COLOR_FORMAT_SWAP)
        .image_color_space(format.color_space)
        .image_extent(surface_caps.current_extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
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

    // Descriptors
    // Pool:
    let pool_sizes = [vk::DescriptorPoolSizeBuilder::new()
        ._type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(3)];
    let create_info = vk::DescriptorPoolCreateInfoBuilder::new()
        .pool_sizes(&pool_sizes)
        .max_sets(2);
    let descriptor_pool =
        unsafe { device.create_descriptor_pool(&create_info, None, None) }.result()?;

    // Layout:
    let bindings = [
        vk::DescriptorSetLayoutBindingBuilder::new() // Swapchain image
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBindingBuilder::new() // Read image
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBindingBuilder::new() // Write image
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];

    let create_info = vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);

    let descriptor_set_layout =
        unsafe { device.create_descriptor_set_layout(&create_info, None, None) }.result()?;

    // Set:
    let descriptor_set_layouts = [descriptor_set_layout];
    let create_info = vk::DescriptorSetAllocateInfoBuilder::new()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&descriptor_set_layouts);

    let descriptor_set = unsafe { device.allocate_descriptor_sets(&create_info) }.result()?[0];

    // Load shader
    let shader_spirv = std::fs::read("shaders/gol.comp.spv").context("Shader failed to load")?;
    let shader_decoded = decode_spv(&shader_spirv).context("Shader decode failed")?;
    let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&shader_decoded);
    let shader_module =
        unsafe { device.create_shader_module(&create_info, None, None) }.result()?;

    // Pipeline
    let push_constant_ranges = [
        vk::PushConstantRangeBuilder::new()
            .offset(0)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(2 * std::mem::size_of::<i32>() as u32)
    ];

    let create_info =
        vk::PipelineLayoutCreateInfoBuilder::new()
        .push_constant_ranges(&push_constant_ranges)
        .set_layouts(&descriptor_set_layouts);
    let pipeline_layout =
        unsafe { device.create_pipeline_layout(&create_info, None, None) }.result()?;

    let entry_point = CString::new("main")?;
    let stage = vk::PipelineShaderStageCreateInfoBuilder::new()
        .stage(vk::ShaderStageFlagBits::COMPUTE)
        .module(shader_module)
        .name(&entry_point)
        .build();
    let create_info = vk::ComputePipelineCreateInfoBuilder::new()
        .stage(stage)
        .layout(pipeline_layout);
    let pipeline =
        unsafe { device.create_compute_pipelines(None, &[create_info], None) }.result()?[0];

    // Create GOL images
    let extent_3d = vk::Extent3DBuilder::new()
        .width(surface_caps.current_extent.width)
        .height(surface_caps.current_extent.height)
        .depth(1)
        .build();
    let create_info = vk::ImageCreateInfoBuilder::new()
        .image_type(vk::ImageType::_2D)
        .extent(extent_3d)
        .mip_levels(1)
        .array_layers(2)
        .format(DATA_FORMAT)
        .tiling(vk::ImageTiling::LINEAR)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::STORAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlagBits::_1);
    let gol_image = unsafe { device.create_image(&create_info, None, None) }.result()?;
    let _gol_image_mem = allocator
        .allocate(&device, gol_image, MemoryTypeFinder::dynamic())
        .result()?;

    // Create intermediate image
    let extent_3d = vk::Extent3DBuilder::new()
        .width(surface_caps.current_extent.width)
        .height(surface_caps.current_extent.height)
        .depth(1)
        .build();
    let create_info = vk::ImageCreateInfoBuilder::new()
        .image_type(vk::ImageType::_2D)
        .extent(extent_3d)
        .mip_levels(1)
        .array_layers(1)
        .format(COLOR_FORMAT_INTERMEDIATE)
        .tiling(vk::ImageTiling::LINEAR)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlagBits::_1);
    let intermediate_image = unsafe { device.create_image(&create_info, None, None) }.result()?;
    let _intermediate_image_mem = allocator
        .allocate(&device, intermediate_image, MemoryTypeFinder::dynamic())
        .result()?;

    // Create image views
    // One for each layer of the GOL image, and one for each swapchain image
    let sub = vk::ImageSubresourceRangeBuilder::new()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
        .build();

    let rgba_cm = vk::ComponentMapping {
        r: vk::ComponentSwizzle::B,
        g: vk::ComponentSwizzle::G,
        b: vk::ComponentSwizzle::R,
        a: vk::ComponentSwizzle::IDENTITY,
    };

    // Create intermediate image view
    let create_info = vk::ImageViewCreateInfoBuilder::new()
        .image(intermediate_image)
        .view_type(vk::ImageViewType::_2D)
        .format(COLOR_FORMAT_INTERMEDIATE)
        .components(rgba_cm)
        .subresource_range(sub);
    let intermediate_image_view =
        unsafe { device.create_image_view(&create_info, None, None) }.result()?;

    // Create game of life image views
    let data_cm = vk::ComponentMapping {
        r: vk::ComponentSwizzle::IDENTITY,
        g: vk::ComponentSwizzle::IDENTITY,
        b: vk::ComponentSwizzle::IDENTITY,
        a: vk::ComponentSwizzle::IDENTITY,
    };

    // Create view A:
    let data_sub = vk::ImageSubresourceRangeBuilder::new()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
        .build();

    let create_info = vk::ImageViewCreateInfoBuilder::new()
        .image(gol_image)
        .view_type(vk::ImageViewType::_2D)
        .format(DATA_FORMAT)
        .components(data_cm)
        .subresource_range(data_sub);

    let gol_image_view_a =
        unsafe { device.create_image_view(&create_info, None, None) }.result()?;

    // Create view B:
    let data_sub = vk::ImageSubresourceRangeBuilder::new()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(1) // Select layer B
        .layer_count(1)
        .build();

    let create_info = vk::ImageViewCreateInfoBuilder::new()
        .image(gol_image)
        .view_type(vk::ImageViewType::_2D)
        .format(DATA_FORMAT)
        .components(data_cm)
        .subresource_range(data_sub);

    let gol_image_view_b =
        unsafe { device.create_image_view(&create_info, None, None) }.result()?;

    // Create synchronization primitives
    let create_info = vk::SemaphoreCreateInfoBuilder::new();

    // Whether or not the frame at this index is available (gpu-only)
    let image_available = unsafe { device.create_semaphore(&create_info, None, None) }.result()?;

    // Whether or not the frame at this index is finished rendering (gpu-only)
    let render_finished = unsafe { device.create_semaphore(&create_info, None, None) }.result()?;

    let mut read_a = false;

    // Main loop
    let mut draw = move |mouse_pos: (i32, i32)| -> Result<()> {
        // Get the index of the next swapchain image,
        // and set up a semaphore to be notified when it is ready.
        let image_index = unsafe {
            device.acquire_next_image_khr(swapchain, u64::MAX, Some(image_available), None, None)
        }
        .result()?;

        let swapchain_image = swapchain_images[image_index as usize];

        // Update descriptor set to include the buffer
        unsafe {
            let intermediate_di = [vk::DescriptorImageInfoBuilder::new()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(intermediate_image_view)];
            let intermediate_desc = vk::WriteDescriptorSetBuilder::new()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&intermediate_di);

            let read_di = [vk::DescriptorImageInfoBuilder::new()
                .image_view(if read_a {
                    gol_image_view_a
                } else {
                    gol_image_view_b
                })
                .image_layout(vk::ImageLayout::GENERAL)];
            let read_desc = vk::WriteDescriptorSetBuilder::new()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&read_di);

            let write_di = [vk::DescriptorImageInfoBuilder::new()
                .image_view(if read_a {
                    gol_image_view_b
                } else {
                    gol_image_view_a
                })
                .image_layout(vk::ImageLayout::GENERAL)];
            let write_desc = vk::WriteDescriptorSetBuilder::new()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&write_di);

            device.update_descriptor_sets(&[intermediate_desc, read_desc, write_desc], &[])
        };

        // Build command buffer
        let command_buffer = command_buffer;
        unsafe {
            device.reset_command_buffer(command_buffer, None).result()?;

            let begin_info = vk::CommandBufferBeginInfoBuilder::new();
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .result()?;

            // Barrier (intermediate `UNDEFINED` -> `GENERAL`)
            let sub = vk::ImageSubresourceRangeBuilder::new()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build();

            let intermediate_image_barrier = vk::ImageMemoryBarrierBuilder::new()
                .image(intermediate_image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .subresource_range(sub);

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                None,
                &[],
                &[],
                &[intermediate_image_barrier],
            );

            // Push constants for mouse pos
            let values = [mouse_pos.0, mouse_pos.1];
            device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                2 * std::mem::size_of::<i32>() as u32,
                &values as *const i32 as _
            );

            // Dispatch
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            const LOCAL_SIZE_X: u32 = 16; // TODO: Better handling of sizes
            const LOCAL_SIZE_Y: u32 = 16;

            device.cmd_dispatch(
                command_buffer,
                surface_caps.current_extent.width / LOCAL_SIZE_X,
                surface_caps.current_extent.height / LOCAL_SIZE_Y,
                1,
            );

            // Barrier (intermediate `GENERAL` -> `TRANSFER_SRC`), (swapchain `UNDEFINED` -> `TRANSFER_DST`)
            let intermediate_image_barrier = vk::ImageMemoryBarrierBuilder::new()
                .image(intermediate_image)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .subresource_range(sub);

            let swapchain_image_barrier = vk::ImageMemoryBarrierBuilder::new()
                .image(swapchain_image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .subresource_range(sub);

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                None,
                &[],
                &[],
                &[intermediate_image_barrier, swapchain_image_barrier],
            );

            // Copy intermediate image to swapchain image
            let off = vk::Offset3DBuilder::new().x(0).y(0).z(0).build();
            let sub_layers = vk::ImageSubresourceLayersBuilder::new()
                .layer_count(1)
                .base_array_layer(0)
                .mip_level(0)
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .build();
            let copy_info = [vk::ImageCopyBuilder::new()
                .src_subresource(sub_layers)
                .src_offset(off)
                .dst_subresource(sub_layers)
                .dst_offset(off)
                .extent(extent_3d)];
            device.cmd_copy_image(
                command_buffer,
                intermediate_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy_info,
            );

            // Barrier (swapchain `TRANSFER_DST` -> `PRESENT_SRC`)
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

            // End command buffer
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
        unsafe { device.queue_submit(queue, &[submit_info], None) }.result()?;

        let swapchains = [swapchain];
        let image_indices = [image_index];
        let present_info = khr_swapchain::PresentInfoKHRBuilder::new()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe { device.queue_present_khr(queue, &present_info) }.result()?;

        unsafe {
            device.queue_wait_idle(queue).result()?;
        }
        read_a = !read_a;
        Ok(())
    };

    let mut mouse_pos = None;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                let PhysicalPosition { x, y, .. } = position;
                mouse_pos = Some((x as i32, y as i32));
            },
            Event::MainEventsCleared => {
                draw(mouse_pos.unwrap_or((-1, -1))).unwrap();
            }
            _ => (),
        }
    });
}
