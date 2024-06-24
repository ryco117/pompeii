/// The types of present modes available to the CLI.
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum PresentMode {
    Immediate,
    TripleBuffer,
    #[default]
    DoubleBuffer,
    DoubleBufferRelaxed,
}
impl From<PresentMode> for ash::vk::PresentModeKHR {
    /// Convert the CLI present mode to the Vulkan equivalent.
    fn from(present_mode: PresentMode) -> Self {
        match present_mode {
            PresentMode::Immediate => ash::vk::PresentModeKHR::IMMEDIATE,
            PresentMode::TripleBuffer => ash::vk::PresentModeKHR::MAILBOX,
            PresentMode::DoubleBuffer => ash::vk::PresentModeKHR::FIFO,
            PresentMode::DoubleBufferRelaxed => ash::vk::PresentModeKHR::FIFO_RELAXED,
        }
    }
}

/// The types of MSAA available to the CLI.
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum MultiSamplingMode {
    #[default]
    Disabled,
    X2,
    X4,
    X8,
    X16,
    X32,
    X64,
}
impl From<MultiSamplingMode> for ash::vk::SampleCountFlags {
    /// Convert the CLI multisampling mode to the Vulkan equivalent.
    fn from(samples: MultiSamplingMode) -> Self {
        match samples {
            MultiSamplingMode::Disabled => ash::vk::SampleCountFlags::TYPE_1,
            MultiSamplingMode::X2 => ash::vk::SampleCountFlags::TYPE_2,
            MultiSamplingMode::X4 => ash::vk::SampleCountFlags::TYPE_4,
            MultiSamplingMode::X8 => ash::vk::SampleCountFlags::TYPE_8,
            MultiSamplingMode::X16 => ash::vk::SampleCountFlags::TYPE_16,
            MultiSamplingMode::X32 => ash::vk::SampleCountFlags::TYPE_32,
            MultiSamplingMode::X64 => ash::vk::SampleCountFlags::TYPE_64,
        }
    }
}

/// The command-line interface for Pompeii.
#[derive(clap::Parser)]
pub struct Args {
    /// The preferred present mode to use. Immediate and double-buffer relaxed may show screen tearing.
    #[arg(short, long, default_value_t, value_enum)]
    pub present_mode: PresentMode,

    /// The type of multisampling to perform, if any.
    #[arg(long, default_value_t, value_enum)]
    pub msaa: MultiSamplingMode,

    /// Prefer presenting to an HDR colorspace if available.
    #[arg(long, default_value_t)]
    pub hdr: bool,
}
