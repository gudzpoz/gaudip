//! This module contains a Gtk widget for displaying Emacs "windows".
//!
//! Note that Emacs windows are not OS windows, but rather panes for
//! displaying buffer contents.

use gtk::PolicyType;
use gtk::glib::Object;
use gtk::prelude::WidgetExt;
use gtk::subclass::prelude::ObjectSubclassIsExt;
use crate::window::text_view::ScrolledTextView;

mod text_view;

glib::wrapper! {
    pub struct BufferWindow(ObjectSubclass<imp::BufferWindow>)
    @extends gtk::Widget,
    @implements gtk::Accessible, gtk::Buildable, gtk::ConstraintTarget;
}
impl BufferWindow {
    pub fn new() -> Self {
        let obj: Self = Object::builder().build();
        obj.set_hexpand(true);
        obj.set_vexpand(true);
        let main = gtk::ScrolledWindow::builder()
            .child(&ScrolledTextView::new())
            .vscrollbar_policy(PolicyType::Always)
            .hscrollbar_policy(PolicyType::Automatic)
            .vexpand(true)
            .hexpand(true)
            .build();
        main.set_parent(&obj);
        let layout = gtk::BoxLayout::builder().build();
        obj.set_layout_manager(Some(layout));
        obj.imp().0.replace(Some(main));
        obj
    }
}
mod imp {
    use std::cell::Cell;
    use std::option::Option;
    use gtk::prelude::WidgetExt;
    use gtk::subclass::prelude::{ObjectImpl, ObjectSubclass, WidgetImpl};
    use gtk::glib;

    #[derive(Default)]
    pub struct BufferWindow(pub(super) Cell<Option<gtk::ScrolledWindow>>);

    #[glib::object_subclass]
    impl ObjectSubclass for BufferWindow {
        const NAME: &'static str = "BufferWindow";
        type Type = super::BufferWindow;
        type ParentType = gtk::Widget;
    }

    impl ObjectImpl for BufferWindow {
        fn dispose(&self) {
            if let Some(child) = self.0.take() {
                child.unparent();
            }
        }
    }
    impl WidgetImpl for BufferWindow {
    }
}
