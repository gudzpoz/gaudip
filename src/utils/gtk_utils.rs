use glib::object::Cast;
use gtk::prelude::WidgetExt;

pub struct GtkChildrenIter(Option<gtk::Widget>);

pub fn iter_children<T: gtk::prelude::IsA<gtk::Widget>>(parent: &T) -> GtkChildrenIter {
    let widget = parent.clone().upcast();
    GtkChildrenIter(widget.first_child())
}

impl Iterator for GtkChildrenIter {
    type Item = gtk::Widget;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.0.take();
        if let Some(child) = &result {
            self.0 = child.next_sibling();
        }
        result
    }
}
