pub trait Float:
    num_traits::float::Float
    + cgmath::BaseFloat
    + numpy::Element
    + std::fmt::Debug
    + Default
    + Send
    + Sync
{
    type BitType: Copy + Clone + PartialEq + Eq + std::fmt::Debug + Send + Sync + std::hash::Hash;

    fn to_bits(self) -> Self::BitType;
    fn from_bits(bits: Self::BitType) -> Self;
}

impl Float for f32 {
    type BitType = u32;

    fn to_bits(self) -> Self::BitType {
        f32::to_bits(self)
    }

    fn from_bits(bits: Self::BitType) -> Self {
        f32::from_bits(bits)
    }
}

impl Float for f64 {
    type BitType = u64;

    fn to_bits(self) -> Self::BitType {
        f64::to_bits(self)
    }

    fn from_bits(bits: Self::BitType) -> Self {
        f64::from_bits(bits)
    }
}
