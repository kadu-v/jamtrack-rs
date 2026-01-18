use nalgebra::Matrix1x4;
use num::Float;
use std::fmt::Debug;

/* ------------------------------------------------------------------------------
 * Type aliases
 * ------------------------------------------------------------------------------ */
pub type Xyah<T> = Matrix1x4<T>;

/* ------------------------------------------------------------------------------
 * Rect struct
 * ------------------------------------------------------------------------------ */
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rect<T>
where
    T: Debug + Float,
{
    tlwh: Matrix1x4<T>,
}

impl<T> Rect<T>
where
    T: Clone + Debug + Float,
{
    pub fn new(x: T, y: T, width: T, height: T) -> Self {
        let tlwh =
            Matrix1x4::new(x.clone(), y.clone(), width.clone(), height.clone());
        Self { tlwh }
    }

    #[inline(always)]
    pub fn x(&self) -> T {
        self.tlwh[(0, 0)]
    }

    #[inline(always)]
    pub fn set_x(&mut self, x: T) {
        self.tlwh[(0, 0)] = x;
    }

    #[inline(always)]
    pub fn y(&self) -> T {
        self.tlwh[(0, 1)]
    }

    #[inline(always)]
    pub fn set_y(&mut self, y: T) {
        self.tlwh[(0, 1)] = y;
    }

    #[inline(always)]
    pub fn width(&self) -> T {
        self.tlwh[(0, 2)]
    }

    #[inline(always)]
    pub fn set_width(&mut self, width: T) {
        self.tlwh[(0, 2)] = width;
    }

    #[inline(always)]
    pub fn height(&self) -> T {
        self.tlwh[(0, 3)]
    }

    #[inline(always)]
    pub fn set_height(&mut self, height: T) {
        self.tlwh[(0, 3)] = height;
    }

    pub fn area(&self) -> T {
        (self.tlwh[(0, 2)] + T::from(1).unwrap())
            * (self.tlwh[(0, 3)] + T::from(1).unwrap())
    }

    pub fn calc_iou(&self, other: &Rect<T>) -> T {
        let box_area = other.area();
        let iw = (self.tlwh[(0, 0)] + self.tlwh[(0, 2)])
            .min(other.tlwh[(0, 0)] + other.tlwh[(0, 2)])
            - (self.tlwh[(0, 0)]).max(other.tlwh[(0, 0)])
            + T::from(1).unwrap();

        let mut iou = T::from(0).unwrap();
        if iw > T::from(0).unwrap() {
            let ih = (self.tlwh[(0, 1)] + self.tlwh[(0, 3)])
                .min(other.tlwh[(0, 1)] + other.tlwh[(0, 3)])
                - (self.tlwh[(0, 1)]).max(other.tlwh[(0, 1)])
                + T::from(1).unwrap();

            if ih > T::from(0).unwrap() {
                let ua = (self.tlwh[(0, 2)] + T::from(1).unwrap())
                    * (self.tlwh[(0, 3)] + T::from(1).unwrap())
                    + box_area
                    - iw * ih;
                iou = iw * ih / ua;
            }
        }
        iou
    }

    pub fn get_xyah(&self) -> Xyah<T> {
        Matrix1x4::new(
            self.tlwh[(0, 0)] + self.tlwh[(0, 2)] / T::from(2).unwrap(),
            self.tlwh[(0, 1)] + self.tlwh[(0, 3)] / T::from(2).unwrap(),
            self.tlwh[(0, 2)] / self.tlwh[(0, 3)],
            self.tlwh[(0, 3)],
        )
    }

    /// Get bounding box as [x1, y1, x2, y2] format
    pub fn get_xyxy(&self) -> [T; 4] {
        [
            self.tlwh[(0, 0)],
            self.tlwh[(0, 1)],
            self.tlwh[(0, 0)] + self.tlwh[(0, 2)],
            self.tlwh[(0, 1)] + self.tlwh[(0, 3)],
        ]
    }

    /// Create Rect from [x1, y1, x2, y2] format
    pub fn from_xyxy(x1: T, y1: T, x2: T, y2: T) -> Self {
        Self::new(x1, y1, x2 - x1, y2 - y1)
    }

    /// Create Rect from [x, y, h, r] format (center_x, center_y, height, aspect_ratio)
    pub fn from_xyhr(x: T, y: T, h: T, r: T) -> Self {
        let w = if r <= T::zero() { T::zero() } else { r * h };
        Self::new(
            x - w / T::from(2).unwrap(),
            y - h / T::from(2).unwrap(),
            w,
            h,
        )
    }
}
