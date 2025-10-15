use super::segment::EStrSegment;
use gtk::pango::AttrList;
use roperig::metrics::{BaseMetric, CharMetric};
use roperig::roperig::Rope;
use roperig::string::RopeContainer;
use std::cell::RefCell;
use crate::utils::pango_utils::{PangoUnit, PixelUnit};

/// A buffer for a line
///
/// All public functions use char indices (rather than byte indices).
#[derive(Default)]
pub struct LineBuffer {
    pango_str: Vec<u8>,
    pango_attrs: AttrList,
    pango_layout: RefCell<Option<gtk::pango::Layout>>,
    metrics: Rope<EStrSegment>,
}

impl RopeContainer<EStrSegment> for LineBuffer {
    fn rope(&self) -> &Rope<EStrSegment> {
        &self.metrics
    }
    fn rope_mut(&mut self) -> &mut Rope<EStrSegment> {
        &mut self.metrics
    }
}

impl LineBuffer {
    pub fn pango_bytes(&self) -> &[u8] {
        &self.pango_str
    }
    pub fn attrs(&self) -> &AttrList {
        &self.pango_attrs
    }
    pub fn pango_layout<F>(&self, context: &gtk::pango::Context, width: i32, mut f: F)
    where F: FnMut(&gtk::pango::Layout) {
        let width = PixelUnit(width);
        let mut inner = self.pango_layout.borrow_mut();
        let layout = match &*inner {
            Some(layout) => {
                if PangoUnit(layout.width()) != width.into() {
                    layout.set_width(PangoUnit::from(width).0);
                }
                layout
            }
            _ => {
                let layout = gtk::pango::Layout::new(context);
                layout.set_width(PangoUnit::from(width).0);
                layout.set_text(&unsafe {
                    str::from_utf8_unchecked(&self.pango_str)
                });
                inner.replace(layout);
                &inner.as_ref().unwrap()
            }
        };
        f(layout);
    }

    pub fn split(&mut self, index: usize) -> LineBuffer {
        let mut pieces = Vec::new();
        let start = self.index_to_byte(index);
        let end = self.metrics.len();
        self.metrics.for_range::<BaseMetric>(start..end, |_, piece, range, _| {
            let mut newp = piece.clone();
            if range.end.bytes != piece.chars_bytes().1 {
                newp.split(range.end.bytes);
            }
            if range.start.bytes != 0 {
                newp = newp.split(range.start.bytes);
            }
            pieces.push(newp);
            true
        });
        let mut new: Rope<EStrSegment> = Rope::default();
        new.insert_many_after(None, pieces.len(), pieces.into_iter());
        let new_line = LineBuffer {
            pango_str: self.pango_str[start..].to_vec(),
            pango_attrs: self.pango_attrs.copy().inspect(|list| {
                list.update(0, start as i32, 0);
            }).unwrap_or_else(|| {
                let new_list = AttrList::new();
                new_list.update(0, 0, (end - start) as i32);
                new_list
            }),
            pango_layout: Default::default(),
            metrics: new,
        };
        self.metrics.delete(start..end - start);
        self.pango_str.resize(start, 0);
        self.pango_attrs.update(start as i32, (end - start) as i32, 0);
        new_line
    }

    pub fn clear(&mut self) {
        self.pango_str.clear();
        self.pango_attrs = AttrList::new();
        self.metrics = Rope::default();
    }

    pub fn delete(&mut self, from: usize, len: usize) {
        if len == 0 {
            return;
        }
        let start = self.index_to_byte(from);
        let end = self.index_to_byte(from + len);
        self.metrics.delete(start..end);
        self.pango_str.drain(start..end);
        self.pango_attrs.update(
            i32::try_from(start).unwrap(),
            i32::try_from(end - start).unwrap(),
            0,
        );
    }

    pub fn insert_ascii(&mut self, index: usize, bytes: &[u8]) {
        self.insert(index, EStrSegment::from_ascii(bytes));
    }
    pub fn insert_raw(&mut self, index: usize, bytes: &[u8]) {
        self.insert(index, EStrSegment::from_raw(bytes));
    }
    pub fn insert_utf32(&mut self, index: usize, bytes: &[u8], stride: usize) {
        self.insert(index, match stride {
            0 => EStrSegment::from_utf32_stride1(bytes),
            1 => EStrSegment::from_utf32_stride2(bytes),
            4 => EStrSegment::from_utf32_stride4(bytes),
            _ => panic!("invalid stride"),
        });
    }
    pub fn insert_emacs(&mut self, index: usize, bytes: &[u8]) {
        let mut vec = Vec::new();
        let mut extra = 0;
        for segment in EStrSegment::from_emacs(bytes) {
            extra += segment.len();
            vec.push(segment);
        }
        let mut offset = self.index_to_byte(index);
        self.expand_str(offset, extra);
        for segment in EStrSegment::from_emacs(bytes) {
            let len = segment.len();
            segment.write(&mut self.pango_str[offset..(offset + len)]);
            offset += len;
        }
        while let Some(element) = vec.pop() {
            self.metrics.insert(offset, element);
        }
        self.pango_attrs.update(
            i32::try_from(offset).unwrap(),
            0,
            i32::try_from(extra).unwrap(),
        );
    }
    pub(crate) fn insert_segment(&mut self, index: usize, s: EStrSegment) {
        self.insert(index, s);
    }
    fn insert(&mut self, index: usize, s: EStrSegment) {
        let offset = self.index_to_byte(index);
        let extra = s.len();
        self.expand_str(offset, extra);
        s.write(&mut self.pango_str[offset..offset + extra]);
        self.metrics.insert(offset, s);
        self.pango_attrs.update(
            i32::try_from(offset).unwrap(),
            0,
            i32::try_from(extra).unwrap(),
        );
    }

    fn expand_str(&mut self, offset: usize, extra: usize) {
        let end = self.pango_str.len();
        self.pango_str.resize(end + extra, 0);
        self.pango_str.copy_within(offset..end, offset + extra);
    }

    fn index_to_byte(&self, index: usize) -> usize {
        self.metrics.convert_metrics::<CharMetric, BaseMetric>(index).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::segment::metrics::tests::utf32_stride4_to_bytes;

    #[test]
    fn test_emacs_string() {
        let mut buffer = LineBuffer::default();
        let s = "Hello! 你好！ こんにちは！˚˖𓍢🌷✧˚.🎀⋆ \u{10FFFF}";
        let vec: Vec<u32> = s.chars().map(|c| c as u32).collect();
        for _ in 0..2 {
            buffer.clear();
            buffer.insert_emacs(0, utf32_stride4_to_bytes(&vec));
            assert_eq!(s.as_bytes(), buffer.pango_str);
        }
    }

    #[test]
    fn test_split() {
        let mut buffer = LineBuffer::default();
        let s = "Hello!World!";
        buffer.insert_ascii(0, s.as_bytes());
        assert_eq!(s.as_bytes(), buffer.pango_bytes());
        let new_line = buffer.split(6);
        assert_eq!("Hello!".as_bytes(), buffer.pango_bytes());
        assert_eq!("World!".as_bytes(), new_line.pango_bytes());
    }
}
