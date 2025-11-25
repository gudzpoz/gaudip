use gtk::graphene::Rect;
use gtk::pango::Rectangle;

const PANGO_SHIFT: i32 = 10;
const PANGO_SCALE: i32 = 1 << PANGO_SHIFT;

/// Length/coordinate in pixels
#[derive(Default, Copy, Clone, Eq, PartialEq)]
pub struct PixelUnit(pub i32);
/// Length/coordinate in pango units (1024 * pixels)
#[derive(Default, Copy, Clone, Eq, PartialEq)]
pub struct PangoUnit(pub i32);

impl PangoUnit {
    pub fn from_pixels(pixels: f64) -> Self {
        PangoUnit((pixels * PANGO_SCALE as f64) as i32)
    }
    pub fn pixels(self) -> i32 {
        PixelUnit::from(self).0
    }
}

impl From<PixelUnit> for PangoUnit {
    fn from(unit: PixelUnit) -> Self {
        if unit.0 < 0 {
            return PangoUnit(-1);
        }
        PangoUnit(unit.0 << PANGO_SHIFT)
    }
}
impl From<PangoUnit> for PixelUnit {
    fn from(val: PangoUnit) -> Self {
        if val.0 < 0 {
            return PixelUnit(-1);
        }
        let floor = val.0 >> PANGO_SHIFT;
        let round = if val.0 & (1 << (PANGO_SHIFT - 1)) != 0 {
            floor + 1
        } else {
            floor
        };
        PixelUnit(round)
    }
}
impl From<PangoUnit> for f64 {
    fn from(val: PangoUnit) -> Self {
        val.0 as f64 / PANGO_SCALE as f64
    }
}
impl From<PangoUnit> for f32 {
    fn from(val: PangoUnit) -> Self {
        let f: f64 = val.into();
        f as f32
    }
}

/// A point that supports both pango units and pixel units
#[derive(Default, Copy, Clone)]
pub struct PangoPoint {
    pango_x: i32,
    pango_y: i32,
}

impl PangoPoint {
    pub fn new_pango(pango_x: i32, pango_y: i32) -> Self {
        PangoPoint { pango_x, pango_y }
    }
    pub fn new_pixel(x: impl Into<f64>, y: impl Into<f64>) -> Self {
        Self::new_pango(
            (x.into() * PANGO_SCALE as f64) as i32,
            (y.into() * PANGO_SCALE as f64) as i32,
        )
    }
    pub fn from_rect(rectangle: Rectangle) -> (PangoPoint, PangoPoint) {
        (
            PangoPoint::new_pango(rectangle.x(), rectangle.y()),
            PangoPoint::new_pango(rectangle.width(), rectangle.height()),
        )
    }

    pub fn x(&self) -> PangoUnit {
        PangoUnit(self.pango_x)
    }
    pub fn y(&self) -> PangoUnit {
        PangoUnit(self.pango_y)
    }
}

pub fn to_graphene_rect(point: PangoPoint, size: PangoPoint) -> Rect {
    Rect::new(
        (point.pango_x as f64 / PANGO_SCALE as f64) as f32,
        (point.pango_y as f64 / PANGO_SCALE as f64) as f32,
        (size.pango_x as f64 / PANGO_SCALE as f64) as f32,
        (size.pango_y as f64 / PANGO_SCALE as f64) as f32,
    )
}
