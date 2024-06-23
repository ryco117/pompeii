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

/// The command-line interface for Pompeii.
#[derive(clap::Parser)]
pub struct Args {
    /// The preferred present mode to use. Immediate and double-buffer relaxed may show screen tearing.
    #[arg(short, long, default_value_t, value_enum)]
    pub present_mode: PresentMode,

    /// Prefer presenting to an HDR colorspace if available.
    #[arg(long, default_value_t)]
    pub hdr: bool,
}
