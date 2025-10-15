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
        self.queue_draw();
    }
}

pub mod imp {
    use super::*;
    use crate::data::lines::BufferLines;
    use gtk::gdk::RGBA;
    use gtk::graphene::Point;
    use gtk::prelude::{ObjectExt, SnapshotExt};
    use gtk::subclass::prelude::*;
    use gtk::Snapshot;
    use std::cell::{Cell, RefCell};
    use crate::utils::pango_utils::PangoUnit;

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

        pub buffer: BufferLines,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for ScrolledTextView {
        const NAME: &'static str = "ScrolledTextView";
        type Type = super::ScrolledTextView;
        type ParentType = gtk::Widget;
        type Interfaces = (gtk::Scrollable,);

        fn new() -> Self {
            let mut buffer = BufferLines::default();
            buffer.insert(0, "3".into());
            buffer.insert_line(0, "1".into());
            buffer.insert_line(1, "2".into());
            buffer.insert_line(3, "4".into());
            Self {
                vadjustment: Default::default(),
                vadjustment_signal: Default::default(),
                hadjustment: Default::default(),
                hadjustment_signal: Default::default(),
                hscroll_policy: Cell::new(gtk::ScrollablePolicy::Natural),
                vscroll_policy: Cell::new(gtk::ScrollablePolicy::Natural),
                buffer,
            }
        }
    }

    #[glib::derived_properties]
    impl ObjectImpl for ScrolledTextView {}
    impl WidgetImpl for ScrolledTextView {
        fn size_allocate(&self, _width: i32, _height: i32, _baseline: i32) {
            let adj_mut = self.vadjustment.borrow_mut();
            if let Some(adj) = &*adj_mut {
                adj.set_upper(self.buffer.lines() as f64);
            }
            self.obj().queue_draw();
        }
        fn snapshot(&self, snapshot: &Snapshot) {
            let ctx = self.obj().pango_context();
            let mut height: f32 = 0.0;
            self.buffer.for_each_physical(|line| {
                line.pango_layout(&ctx, self.obj().width(), |layout| {
                    snapshot.translate(&Point::new(0.0, height));
                    snapshot.append_layout(&layout, &RGBA::BLACK);
                    height = f32::from(PangoUnit(layout.extents().1.height()));
                });
            });
        }
    }
    impl ScrollableImpl for ScrolledTextView {}

    macro_rules! define_adjustment_setter { ($name:ident, $field:ident, $signal:ident) => {
        fn $name(&self, value: Option<gtk::Adjustment>) {
            let obj = self.obj();
            if let (Some(signal_id), Some(obj)) = (
                self.$signal.take(), self.$field.take(),
            ) {
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
