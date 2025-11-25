use glib::subclass::types::ObjectSubclassIsExt;
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

    pub fn on_adjusted(&self) {
        let imp = self.imp();
        let prev = imp.v_ratio.get();
        let next = imp.vadjustment.borrow().as_ref()
            .map(|adj| adj.value() / adj.upper())
            .unwrap_or(0.0);
        imp.v_ratio.set(next);
        let mut buffer = imp.buffer.borrow_mut();
        buffer.scroll(if next > prev {
            (next - prev) / (1.0 - prev)
        } else {
            (next - prev) / prev
        });
        self.queue_draw();
    }
}

pub mod imp {
    use super::*;
    use crate::data::virtlines::{LineInfo, VirtualLines};
    use gtk::gdk::RGBA;
    use gtk::graphene::Rect;
    use gtk::prelude::{ObjectExt, SnapshotExt};
    use gtk::subclass::prelude::*;
    use gtk::Snapshot;
    use std::cell::{Cell, RefCell};
    use std::num::NonZero;
    use crate::utils::pango_utils::{PangoUnit, PixelUnit};

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

        pub buffer: RefCell<VirtualLines>,
        pub v_ratio: Cell<f64>,
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
                buffer.insert_line(NonZero::new(i + 1).unwrap(), LineInfo {
                    chars: 1, lines: 1, height: PangoUnit::from(PixelUnit(20)).0 as usize,
                }, NonZero::new(i + 1));
            }
/*             buffer.insert(0, "3".into());
            buffer.insert_line(0, "1𒐫 a⃰⃰⃰⃰⃰⃰⃰".into());
            buffer.insert_line(1, "2".into());
            buffer.insert_line(3, "4".into()); */
            Self {
                vadjustment: Default::default(),
                vadjustment_signal: Default::default(),
                hadjustment: Default::default(),
                hadjustment_signal: Default::default(),
                hscroll_policy: Cell::new(gtk::ScrollablePolicy::Natural),
                vscroll_policy: Cell::new(gtk::ScrollablePolicy::Natural),
                buffer: buffer.into(),
                v_ratio: Default::default(),
            }
        }
    }

    #[glib::derived_properties]
    impl ObjectImpl for ScrolledTextView {}
    impl WidgetImpl for ScrolledTextView {
        fn size_allocate(&self, _width: i32, _height: i32, _baseline: i32) {
            let adj_mut = self.vadjustment.borrow_mut();
            if let Some(adj) = &*adj_mut {
                let buffer = self.buffer.borrow();
                adj.set_upper(buffer.height() as f64);
            }
            self.obj().queue_draw();
        }
        fn snapshot(&self, snapshot: &Snapshot) {
            let limit = self.obj().height();
            let buffer = self.buffer.borrow();
            for (pos, i, _line) in buffer.iter_from_current() {
                let y = PangoUnit(pos as i32).into();
                let w = ((i.map(|i| i.get()).unwrap_or(0)) % 10) * 10 + 10;
                snapshot.append_color(&RGBA::GREEN, &Rect::new(0.0, y, w as f32, 10.0));
                if f32::from(PangoUnit(pos as i32)) > limit as f32 {
                    break;
                }
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
                self.$signal.replace(Some(adj.connect_value_changed(glib::clone!(
                    #[weak] obj,
                    move |_value| { obj.on_adjusted() },
                ))));
                obj.queue_allocate();
                obj.on_adjusted();
            }
            self.$field.replace(value);
        }
    } }
    impl ScrolledTextView {
        define_adjustment_setter!(set_vadjustment, vadjustment, vadjustment_signal);
        define_adjustment_setter!(set_hadjustment, hadjustment, hadjustment_signal);
    }
}
