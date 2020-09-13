use anyhow::Result;
use erupt::{
    extensions::khr_surface,
    vk1_0 as vk, InstanceLoader,
};
use std::{
    ffi::CStr,
    os::raw::c_char,
};

#[derive(Debug)]
pub struct HardwareSelection {
    pub physical_device: vk::PhysicalDevice,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
    pub queue_family: u32,
    pub format: khr_surface::SurfaceFormatKHR,
    pub present_mode: khr_surface::PresentModeKHR,
}

// TODO: Flatten this and replace .unwrap() with .result()?
impl HardwareSelection {
    pub fn query(
        instance: &InstanceLoader,
        surface: khr_surface::SurfaceKHR,
        device_extensions: &[*const c_char],
    ) -> Result<Self> {
        unsafe { instance.enumerate_physical_devices(None) }
            .unwrap()
            .into_iter()
            .filter_map(|physical_device| unsafe {
                let queue_family = match instance
                    .get_physical_device_queue_family_properties(physical_device, None)
                    .into_iter()
                    .enumerate()
                    .position(|(i, properties)| {
                        properties.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                            && instance
                                .get_physical_device_surface_support_khr(
                                    physical_device,
                                    i as u32,
                                    surface,
                                    None,
                                )
                                .unwrap()
                                == true
                    }) {
                    Some(queue_family) => queue_family as u32,
                    None => return None,
                };

                let formats = instance
                    .get_physical_device_surface_formats_khr(physical_device, surface, None)
                    .unwrap();
                let format = match formats
                    .iter()
                    .find(|surface_format| {
                        surface_format.format == vk::Format::B8G8R8A8_SRGB
                            && surface_format.color_space
                                == khr_surface::ColorSpaceKHR::SRGB_NONLINEAR_KHR
                    })
                    .or_else(|| formats.get(0))
                {
                    Some(surface_format) => surface_format.clone(),
                    None => return None,
                };

                let present_mode = instance
                    .get_physical_device_surface_present_modes_khr(physical_device, surface, None)
                    .unwrap()
                    .into_iter()
                    .find(|present_mode| present_mode == &khr_surface::PresentModeKHR::MAILBOX_KHR)
                    .unwrap_or(khr_surface::PresentModeKHR::FIFO_KHR);

                let supported_extensions = instance
                    .enumerate_device_extension_properties(physical_device, None, None)
                    .unwrap();
                if !device_extensions.iter().all(|device_extension| {
                    let device_extension = CStr::from_ptr(*device_extension);

                    supported_extensions.iter().any(|properties| {
                        CStr::from_ptr(properties.extension_name.as_ptr()) == device_extension
                    })
                }) {
                    return None;
                }

                let physical_device_properties =
                    instance.get_physical_device_properties(physical_device, None);
                Some(Self {
                    physical_device,
                    queue_family,
                    format,
                    present_mode,
                    physical_device_properties,
                })
            })
            .max_by_key(|query| match query.physical_device_properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 2,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 0,
            })
            .ok_or(anyhow::format_err!("No suitable hardware found for this configuration"))
    }
}
