use gtk::prelude::{AdjustmentExt, WidgetExt};

glib::wrapper! {
    pub struct ScrolledTextView(ObjectSubclass<imp::ScrolledTextView>)
    @extends gtk::Widget,
    @implements gtk::Accessible, gtk::Buildable, gtk::ConstraintTarget, gtk::Scrollable;
}
impl ScrolledTextView {
    pub fn new() -> Self {
        let new: Self = glib::Object::builder().build();
        new.set_hexpand(true);
        new.set_vexpand(true);
        new
    }
}

pub mod imp {
    use super::*;
    use crate::data::buffer::LineBuffer;
    use crate::data::virtlines::{LineInfo, VirtualLines};
    use crate::utils::pango_utils::{PangoUnit, PixelUnit};
    use glib;
    use glib::clone;
    use gtk::gdk::RGBA;
    use gtk::graphene::Point;
    use gtk::pango::{Layout, Rectangle, WrapMode};
    use gtk::prelude::{ObjectExt, SnapshotExt};
    use gtk::subclass::prelude::*;
    use gtk::{EventControllerMotion, Snapshot};
    use std::cell::{Cell, RefCell};
    use std::num::NonZero;

    #[derive(glib::Properties)]
    #[properties(wrapper_type = super::ScrolledTextView)]
    pub struct ScrolledTextView {
        #[property(name = "vadjustment",get, set = Self::set_vadjustment, override_interface = gtk::Scrollable)]
        pub vadjustment: RefCell<Option<gtk::Adjustment>>,
        pub vadjustment_signal: RefCell<Option<glib::SignalHandlerId>>,
        #[property(get, set = Self::set_hadjustment, override_interface = gtk::Scrollable)]
        pub hadjustment: RefCell<Option<gtk::Adjustment>>,
        pub hadjustment_signal: RefCell<Option<glib::SignalHandlerId>>,
        #[property(get, set, override_interface = gtk::Scrollable)]
        pub hscroll_policy: Cell<gtk::ScrollablePolicy>,
        #[property(get, set, override_interface = gtk::Scrollable)]
        pub vscroll_policy: Cell<gtk::ScrollablePolicy>,

        view: RefCell<VirtualLines<LineInner>>,
        v_ratio: Cell<f64>,
    }

    struct LineInner {
        text: LineBuffer,
        rendered: RefCell<Option<Layout>>,
        rendered_width: Cell<i32>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for ScrolledTextView {
        const NAME: &'static str = "ScrolledTextView";
        type Type = super::ScrolledTextView;
        type ParentType = gtk::Widget;
        type Interfaces = (gtk::Scrollable,);

        fn new() -> Self {
            let mut buffer = VirtualLines::default();
            for i in 0..100 {
                let mut text = LineBuffer::default();
                // TODO: this is test texts (emoji, large glyphs, grapheme clusters, etc.)
                text.insert(0, format!("Text Line 🥝 𒐫 a⃰⃰⃰⃰⃰⃰⃰ {}", i).as_str().into());
                let text = LineInner {
                    text, rendered: Default::default(), rendered_width: Default::default(),
                };
                buffer.insert_line(NonZero::new(i + 1).unwrap(), LineInfo {
                    chars: text.text.chars() + 1, // +LF
                    lines: 1,
                    height: PangoUnit::from(PixelUnit(2000)).0 as usize,
                }, Some(text));
            }
            Self {
                vadjustment: Default::default(),
                vadjustment_signal: Default::default(),
                hadjustment: Default::default(),
                hadjustment_signal: Default::default(),
                hscroll_policy: Cell::new(gtk::ScrollablePolicy::Natural),
                vscroll_policy: Cell::new(gtk::ScrollablePolicy::Natural),
                view: buffer.into(),
                v_ratio: Default::default(),
            }
        }
    }

    #[glib::derived_properties]
    impl ObjectImpl for ScrolledTextView {
        fn constructed(&self) {
            self.parent_constructed();

            let motion = EventControllerMotion::new();
            let this = self;
            motion.connect_motion(clone!(
                #[weak] this,
                move |_, x, y| {
                    this.on_hover(x, y);
                }
            ));
            self.obj().add_controller(motion);
        }
    }
    impl WidgetImpl for ScrolledTextView {
        fn size_allocate(&self, _width: i32, height: i32, _baseline: i32) {
            let buffer = self.view.borrow();
            self.update_vscroll(height as usize, buffer.height());
            self.obj().queue_draw();
        }
        fn snapshot(&self, snapshot: &Snapshot) {
            let obj = self.obj();
            let width = PangoUnit::from(PixelUnit(obj.width())).0;
            let limit = PangoUnit::from(PixelUnit(self.obj().height())).0 as isize;
            let mut buffer = self.view.borrow_mut();
            let mut iter = buffer.iter_from_current();
            let mut offset = 0;
            let mut height_updated = false;
            while iter.has_next() && offset < limit {
                let Some((rel_offset, line, info)) = iter.current(&buffer) else { break };
                offset += rel_offset;
                snapshot.translate(&Point::new(0.0, PangoUnit(rel_offset as i32).into()));

                let mut height = info.height;
                if let Some(line) = line {
                    let mut rendered = line.rendered.borrow_mut();
                    let layout = match rendered.as_ref() {
                        Some(layout) => {
                            if line.rendered_width.get() != width {
                                layout.set_width(width);
                                line.rendered_width.set(width);
                            }
                            layout
                        }
                        _ => {
                            let layout = Layout::new(&obj.pango_context());
                            layout.set_single_paragraph_mode(true);
                            layout.set_width(width);
                            layout.set_wrap(WrapMode::WordChar);
                            layout.set_text(str::from_utf8(line.text.pango_bytes()).unwrap_or("\\"));

                            line.rendered_width.set(width);
                            rendered.replace(layout);
                            rendered.as_ref().expect("just set")
                        }
                    };
                    snapshot.append_layout(layout, &RGBA::BLACK);

                    let actual_height = layout.extents().1.height();
                    drop(rendered);

                    if actual_height as usize != height {
                        iter.update_line(&mut buffer, &LineInfo {
                            chars: 0, lines: 0,
                            height: (actual_height as usize).wrapping_sub(height),
                        });
                        height = actual_height as usize;
                        height_updated = true;
                    }
                }
                snapshot.translate(&Point::new(0.0, PangoUnit(height as i32).into()));
                offset = offset.saturating_add_unsigned(height);
                iter.advance(&buffer);
            }
            if height_updated {
                // We should not call update_vscroll in snapshot,
                // so we ask to queue a resize, where size_allocate will call update_vscroll.
                obj.queue_resize();
            }
        }
    }
    impl ScrollableImpl for ScrolledTextView {}

    macro_rules! define_adjustment_setter { ($name:ident, $field:ident, $signal:ident) => {
        fn $name(&self, value: Option<gtk::Adjustment>) {
            let obj = self.obj();
            if let (Some(signal_id), Some(obj)) = (self.$signal.take(), self.$field.take()) {
                obj.disconnect(signal_id);
            }
            if let Some(adj) = &value {
                let this = self;
                self.$signal.replace(Some(adj.connect_value_changed(glib::clone!(
                    #[weak] this,
                    move |_value| { this.on_adjusted() },
                ))));
                obj.queue_allocate();
                this.on_adjusted();
            }
            self.$field.replace(value);
        }
    } }
    impl ScrolledTextView {
        define_adjustment_setter!(set_vadjustment, vadjustment, vadjustment_signal);
        define_adjustment_setter!(set_hadjustment, hadjustment, hadjustment_signal);

        fn update_vscroll(&self, height: usize, total: usize) {
            let adj_mut = self.vadjustment.borrow_mut();
            if let Some(adj) = &*adj_mut {
                adj.set_lower(0.0);
                let page = PangoUnit::from(PixelUnit(height as i32)).0 as f64;
                adj.set_page_size(page);
                adj.set_upper(total as f64 + page);
            }
        }

        fn on_adjusted(&self) {
            let prev = self.v_ratio.get();
            let next = self.vadjustment.borrow().as_ref()
                .map(|adj| adj.value() / adj.upper())
                .unwrap_or(0.0);
            self.v_ratio.set(next);
            let mut buffer = self.view.borrow_mut();
            buffer.scroll(if next > prev {
                (next - prev) / (1.0 - prev)
            } else {
                (next - prev) / prev
            });
            self.obj().queue_draw();
        }

        fn on_hover(&self, x: f64, y: f64) {
            let y = PangoUnit::from_pixels(y).0;
            let mut buffer = self.view.borrow_mut();
            let Some((y, iter)) = buffer.line_at_rel_height(y as usize) else { return };
            let Some(line) = iter.current_mut(&mut buffer) else { return };
            let mut rendered = line.rendered.borrow_mut();
            let Some(layout) = &*rendered else { return };
            let byte_index = find_cursor_position(layout, x, y);
            let chars = line.text.byte_index_to_char(byte_index);
            line.text.insert(chars, ".".into());
            rendered.take();
            drop(rendered);
            iter.update_line(&mut buffer, &LineInfo {
                chars: 1,
                lines: 0,
                height: 0,
            });
            self.obj().queue_draw();
        }
    }

    fn find_cursor_position(layout: &Layout, x_px: f64, y: usize) -> usize {
        let x = PangoUnit::from_pixels(x_px).0;
        let y = y as i32;
        let (_inside, byte_index, remaining) = layout.xy_to_index(x, y);
        let text = layout.text();
        let s = text.as_str();
        let (before_glyph, _) = layout.cursor_pos(byte_index);
        let next_index = str_indices::chars::to_byte_idx(
            &s[byte_index as usize..], (remaining as usize).max(1),
        ) as i32 + byte_index;
        let (after_glyph, _) = layout.cursor_pos(next_index);

        // We use floating point here to avoid integer overflow.
        let y_px = f64::from(PangoUnit(y));
        let dist2p = |rx: f64, ry: f64| {
            let dx = rx - x_px;
            let dy = ry - y_px;
            dx * dx + dy * dy
        };
        let dist2 = |rect: &Rectangle| {
            let rx = rect.x();
            let ry = rect.y();
            let rw = rect.width();
            let rh = rect.height();
            dist2p(PangoUnit(rx + rw / 2).into(), PangoUnit(ry + rh / 2).into())
        };
        if dist2(&before_glyph) < dist2(&after_glyph) {
            byte_index as usize
        } else {
            next_index as usize
        }
    }
}
