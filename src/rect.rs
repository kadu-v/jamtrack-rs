use nalgebra::Matrix1x4;
use num::Float;
use std::fmt::Debug;

/*------------------------------------------------------------------------------
Type aliases
------------------------------------------------------------------------------*/
// type Tlwh<T> = Matrix1x4<T>;

type Xyah<T> = Matrix1x4<T>;

/*------------------------------------------------------------------------------
Rect struct
------------------------------------------------------------------------------*/
#[derive(Debug, Clone)]
pub struct Rect<T>
where
    T: Debug + Float,
{
    pub tlwh: Matrix1x4<T>,
    pub(crate) x: T,
    pub(crate) y: T,
    pub(crate) width: T,
    pub(crate) height: T,
}

impl<T> Rect<T>
where
    T: Clone + Debug + Float,
{
    pub fn new(x: T, y: T, width: T, height: T) -> Self {
        let tlwh =
            Matrix1x4::new(x.clone(), y.clone(), width.clone(), height.clone());
        Self {
            tlwh,
            x,
            y,
            width,
            height,
        }
    }

    pub fn x(&self) -> T {
        self.tlwh[(0, 0)]
    }

    pub fn y(&self) -> T {
        self.tlwh[(0, 1)]
    }

    pub fn width(&self) -> T {
        self.tlwh[(0, 2)]
    }

    pub fn height(&self) -> T {
        self.tlwh[(0, 3)]
    }

    pub fn area(&self) -> T {
        (self.width + T::from(1).unwrap()) * (self.height + T::from(1).unwrap())
    }

    pub fn calc_iou(&self, other: &Rect<T>) -> T {
        let box_area = self.area();
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
                let ua = box_area + other.area() - iw * ih;
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
}
