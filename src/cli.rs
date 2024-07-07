/// The types of present modes available to the CLI.
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum PresentMode {
    Immediate,
    Mailbox,
    #[default]
    Fifo,
    FifoRelaxed,
}
impl From<PresentMode> for ash::vk::PresentModeKHR {
    /// Convert the CLI present mode to the Vulkan equivalent.
    fn from(present_mode: PresentMode) -> Self {
        match present_mode {
            PresentMode::Immediate => ash::vk::PresentModeKHR::IMMEDIATE,
            PresentMode::Mailbox => ash::vk::PresentModeKHR::MAILBOX,
            PresentMode::Fifo => ash::vk::PresentModeKHR::FIFO,
            PresentMode::FifoRelaxed => ash::vk::PresentModeKHR::FIFO_RELAXED,
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
    /// The preferred present mode to use.
    /// Immediate and FIFO-relaxed may show screen tearing.
    /// FIFO is the most efficient, and mailbox is the most responsive.
    #[arg(short, long, default_value_t, value_enum)]
    pub present_mode: PresentMode,

    /// The type of multisampling to prefer using, if any. If the desired multi-sample count is not supported, single sampling will be used instead.
    #[arg(long, default_value_t, value_enum)]
    pub msaa: MultiSamplingMode,

    /// Start with FXAA (a fast screen-space anti-aliasing algorithm) enabled.
    #[arg(long, default_value_t)]
    pub fxaa: bool,
}
