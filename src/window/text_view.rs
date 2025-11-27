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
    use crate::events::tokio_runtime;
    use crate::utils::pango_utils::{PangoUnit, PixelUnit};
    use glib;
    use glib::clone;
    use gtk::gdk::{Cursor, RGBA};
    use gtk::graphene::{Point, Rect};
    use gtk::gsk::BlendMode;
    use gtk::pango::{Layout, Rectangle, WrapMode};
    use gtk::prelude::{ObjectExt, SnapshotExt};
    use gtk::subclass::prelude::*;
    use gtk::{EventControllerMotion, Snapshot};
    use tokio::sync::Mutex;
    use std::cell::{Cell, RefCell};
    use std::num::NonZero;
    use std::ops::Range;
    use std::sync::Arc;

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

        cursor: Option<Cursor>,

        remote: Arc<Mutex<MockRpcClient>>,
        change_version: Cell<usize>,
        visit_line: Cell<Option<usize>>,

        view: RefCell<VirtualLines<LineInner>>,
        v_ratio: Cell<f64>,
    }

    const MAX_FETCH_LINES: usize = 512;
    const FETCH_LINE_LOOKAHEAD: usize = 128;

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
            let mock = MockRpcClient();
            buffer.init_lines(&mut Some((mock.summary(), None)).into_iter());
            Self {
                vadjustment: Default::default(),
                vadjustment_signal: Default::default(),
                hadjustment: Default::default(),
                hadjustment_signal: Default::default(),
                hscroll_policy: Cell::new(gtk::ScrollablePolicy::Natural),
                vscroll_policy: Cell::new(gtk::ScrollablePolicy::Natural),
                cursor: Cursor::from_name("text", None),
                remote: Arc::new(Mutex::new(mock)),
                change_version: Default::default(),
                visit_line: Default::default(),
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
            self.update_vscroll(&buffer, height as usize);
            self.obj().queue_draw();
        }
        fn snapshot(&self, snapshot: &Snapshot) {
            self.try_snapshot(Some(snapshot));
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

        fn update_vscroll(&self, buffer: &VirtualLines<LineInner>, height: usize) {
            let adj_mut = self.vadjustment.borrow_mut();
            if let Some(adj) = &*adj_mut {
                let total = buffer.height() as f64;
                adj.set_step_increment(50.0);
                adj.set_lower(0.0);
                let page = height as f64;
                adj.set_page_size(page);
                adj.set_upper(total);
            }
        }

        fn on_adjusted(&self) {
            let prev = self.v_ratio.get();
            let next = self.vadjustment.borrow().as_ref()
                .map(|adj| adj.value() / (adj.upper() - adj.page_size()))
                .unwrap_or(0.0);
            self.v_ratio.set(next);
            let mut buffer = self.view.borrow_mut();
            buffer.scroll(if next > prev {
                (next - prev) / (1.0 - prev)
            } else {
                (next - prev) / prev
            });
            drop(buffer);
            self.try_snapshot(None); // pre-fetch lines to reduce flickering
            self.obj().queue_draw();
        }

        fn on_hover(&self, x: f64, y: f64) {
            let mut buffer = self.view.borrow_mut();
            let Some((y, iter)) = buffer.line_at_rel_height(y as usize) else { return };
            let Some(line_opt) = iter.current_mut(&mut buffer) else { return };
            let Some(line) = line_opt else { return };
            let mut rendered = line.rendered.borrow_mut();
            let Some(layout) = &*rendered else { return };
            let (inside, byte_index) = find_cursor_position(layout, x, y);
            if inside {
                if let Some(cursor) = &self.cursor {
                    self.obj().set_cursor(Some(cursor));
                }
            } else {
                self.obj().set_cursor(None);
            }
            let chars = line.text.byte_index_to_char(byte_index);
            line.text.insert(chars, ".".into());
            rendered.take();
            drop(rendered);
            iter.update_line(&mut buffer, &LineInfo {
                chars: 0, // TODO: call MockRpcClient to update server
                lines: 0,
                height: 0,
            });
            self.change_version.set(self.change_version.get() + 1);
            self.obj().queue_draw();
        }

        fn queue_fetch_task(&self, from: NonZero<usize>, to: NonZero<usize>) {
            let from = from.get();
            let to = to.get();
            if from > to {
                return;
            }

            let obj = self.obj();
            glib::spawn_future_local(clone!(
                #[strong] obj,
                async move {
                    let imp = obj.imp();
                    let version = imp.change_version.get();
                    let remote = imp.remote.clone();
                    let lines = tokio_runtime().spawn(async move {
                        // Simulated delay
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                        let r = remote.lock().await;
                        let to = to.min(from + MAX_FETCH_LINES);
                        r.get_lines(from..to)
                    }).await;
                    if let Ok(lines) = lines {
                        let imp = obj.imp();
                        if version == imp.change_version.get() {
                            imp.update_lines(lines);
                            imp.change_version.set(version + 1);
                            obj.queue_resize();
                        } else {
                            obj.imp().queue_fetch_task(NonZero::new(from).unwrap(), NonZero::new(to).unwrap());
                        }
                    }
                }
            ));
        }

        fn update_lines(&self, lines: Vec<LineResult>) {
            let mut buffer = self.view.borrow_mut();
            let height = buffer.current_height();
            let mut visited = false;
            let visit_line = self.visit_line.get();
            for LineResult { start_char, line_num, line } in lines {
                visited = visited || visit_line == Some(line_num);
                let Some(line_num) = NonZero::new(line_num) else { continue };
                let iter = buffer.iter_from_line(line_num);
                if !iter.has_current() {
                    continue;
                }
                if let Some(Some(_)) = iter.current_mut(&mut buffer) {
                    continue;
                }
                let total_height = buffer.height();
                let total_lines = buffer.line_count();
                let prev_lines = line_num.get() - 1;
                let chars = line.chars() + 1; // "+1" for newline
                iter.materialize(
                    &mut buffer,
                    LineInner {
                        text: line,
                        rendered: Default::default(), rendered_width: Default::default(),
                    },
                    &LineInfo {
                        chars: start_char,
                        lines: prev_lines,
                        height: total_height.checked_div(total_lines).unwrap_or(0) * prev_lines,
                    },
                    LineInfo { chars, lines: 1, height: 20 },
                );
            }
            if visited {
                if let Some(line) = visit_line {
                    if let Some(line) = NonZero::new(line) {
                        buffer.scroll_to_line(line);
                    }
                }
            } else {
                buffer.scroll_to(height);
            }
        }

        fn try_snapshot(&self, snapshot: Option<&Snapshot>) {
            let obj = self.obj();
            let width = PangoUnit::from(PixelUnit(obj.width())).0;
            let limit = self.obj().height() as isize;
            let mut buffer = self.view.borrow_mut();
            let mut iter = buffer.iter_from_current();
            let mut offset = 0;
            let mut height_updated = false;
            let mut fetch_needed_is_first = true;
            let mut fetch_needed_visit = None;
            let mut fetch_needed = None;
            while iter.has_current() && offset < limit {
                let Some((rel_offset, line, info)) = iter.current(&buffer) else { break };
                offset += rel_offset;
                if let Some(s) = snapshot {
                    s.translate(&Point::new(0.0, rel_offset as f32));
                }

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
                    if let Some(s) = snapshot {
                        s.append_layout(layout, &RGBA::BLACK);
                    }

                    let actual_height = PangoUnit(layout.extents().1.height()).pixels();
                    drop(rendered);

                    if actual_height as usize != height {
                        iter.update_line(&mut buffer, &LineInfo {
                            chars: 0, lines: 0,
                            height: (actual_height as usize).wrapping_sub(height),
                        });
                        height = actual_height as usize;
                        height_updated = true;
                    }
                } else if fetch_needed.is_none() {
                    let start_line = iter.current_line_num(&buffer);
                    let line_offset_est = (info.lines * rel_offset.unsigned_abs()).checked_div(info.height).unwrap_or(0);
                    fetch_needed = start_line.and_then(
                        |l| NonZero::new(l.get() + line_offset_est.saturating_sub(FETCH_LINE_LOOKAHEAD)),
                    );
                    if fetch_needed_is_first {
                        fetch_needed_visit = start_line;
                    }
                }
                if let Some(s) = snapshot {
                    s.translate(&Point::new(0.0, height as f32));
                }
                offset = offset.saturating_add_unsigned(height);
                iter.advance(&buffer);
                fetch_needed_is_first = false;
            }
            if let Some(from) = fetch_needed {
                if let Some(to) = iter.current_line_num(&buffer) {
                    self.visit_line.replace(fetch_needed_visit.map(|i| i.get()));
                    self.queue_fetch_task(from, to);
                }
            }
            if height_updated && snapshot.is_some() {
                // We should not call update_vscroll in snapshot,
                // so we ask to queue a resize, where size_allocate will call update_vscroll.
                obj.queue_resize();
            }
        }
    }

    fn find_cursor_position(layout: &Layout, x_px: f64, y: usize) -> (bool, usize) {
        let x = PangoUnit::from_pixels(x_px).0;
        let y = PangoUnit::from_pixels(y as f64).0;
        let (inside, byte_index, remaining) = layout.xy_to_index(x, y);
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
        (inside, if dist2(&before_glyph) < dist2(&after_glyph) {
            byte_index as usize
        } else {
            next_index as usize
        })
    }

    struct LineResult {
        start_char: usize,
        line_num: usize,
        line: LineBuffer,
    }
    struct MockRpcClient();
    const TEST_STR: &str = "Text Line إلا بسم الله 🥝 𒐫  a⃰⃰⃰⃰⃰⃰⃰ ";
    impl MockRpcClient {
        fn summary(&self) -> LineInfo {
            let lines = 10_000_000;
            LineInfo {
                lines,
                chars: (TEST_STR.chars().count() + 10 + 1) * lines,
                height: 20 * lines,
            }
        }
        fn gen_line(&self, i: usize) -> LineBuffer {
            let mut text = LineBuffer::default();
            // TODO: this is test texts (emoji, large glyphs, grapheme clusters, etc.)
            text.insert(0, format!("{}{:010}", TEST_STR, i).as_str().into());
            text
        }
        fn get_lines(&self, lines: Range<usize>) -> Vec<LineResult> {
            lines.map(|i| LineResult {
                start_char: (TEST_STR.chars().count() + 10 + 1) * (i - 1),
                line_num: i,
                line: self.gen_line(i),
            }).collect()
        }
    }
}

