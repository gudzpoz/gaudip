use gtk::glib;
use gtk::glib::Object;
use gtk::prelude::*;
use gtk::{Application, ApplicationWindow};

const APP_ID: &str = "party.iroiro.juicemacs.gaudip";

glib::wrapper! {
    pub struct TestWidget(ObjectSubclass<imp::TestWidget>)
    @extends gtk::Widget,
    @implements gtk::Accessible, gtk::Buildable, gtk::ConstraintTarget;
}
mod imp {
    use gtk::prelude::{SnapshotExt, WidgetExt};
    use gtk::subclass::prelude::{ObjectImpl, ObjectSubclass, ObjectSubclassExt, WidgetImpl};
    use gtk::{gdk, glib, graphene, Snapshot};

    #[derive(Default)]
    pub struct TestWidget();

    #[glib::object_subclass]
    impl ObjectSubclass for TestWidget {
        const NAME: &'static str = "TestWidget";
        type Type = super::TestWidget;
        type ParentType = gtk::Widget;
    }

    impl ObjectImpl for TestWidget {}
    impl WidgetImpl for TestWidget {
        fn snapshot(&self, snapshot: &Snapshot) {
            let red = gdk::RGBA::RED;
            let green = gdk::RGBA::GREEN;
            let yellow = gdk::RGBA::parse("yellow").unwrap();
            let blue = gdk::RGBA::BLUE;
            let obj = self.obj();
            let w: f32 = obj.width() as f32 / 2.0;
            let h: f32 = obj.height() as f32 / 2.0;
            for (i, color) in [red, green, yellow, blue].iter().enumerate() {
                let x = w * (i % 2) as f32;
                let y = h * (i / 2) as f32;
                snapshot.append_color(color, &graphene::Rect::new(x, y, w, h));
            }
        }
    }
}

fn main() -> glib::ExitCode {
    env_logger::init();
    glib::log_set_writer_func(glib_logger);

    let app = Application::builder().application_id(APP_ID).build();
    app.connect_activate(build_ui);
    app.run()
}

fn build_ui(app: &Application) {
    let child: TestWidget = Object::builder().build();

    let window = ApplicationWindow::builder()
        .application(app)
        .title("九叠")
        .child(&child)
        .build();

    window.present();
}

fn glib_logger(level: glib::LogLevel, fields: &[glib::LogField]) -> glib::LogWriterOutput {
    let level = match level {
        glib::LogLevel::Error | glib::LogLevel::Critical => log::Level::Error,
        glib::LogLevel::Warning => log::Level::Warn,
        glib::LogLevel::Message | glib::LogLevel::Info => log::Level::Info,
        glib::LogLevel::Debug => log::Level::Debug,
    };

    let find_field = |name: &str| -> Option<&str> {
        fields.iter().find_map(|field| {
            if field.key() == name {
                field.value_str()
            } else {
                None
            }
        })
    };

    let tag = find_field("GLIB_DOMAIN").unwrap_or("<null>");
    let message = find_field("MESSAGE");
    let file = find_field("CODE_FILE");
    let line = find_field("CODE_LINE");
    let func = find_field("CODE_FUNC");

    if let Some(message) = message {
        if let Some(file) = file {
            let line = line.unwrap_or("0");
            let func = func.unwrap_or("<unknown>");
            log::log!(target: tag, level, "{}:{}:{}: {}", file, line, func, message);
        } else {
            log::log!(target: tag, level, "{}", message);
        }
        glib::LogWriterOutput::Handled
    } else {
        glib::LogWriterOutput::Unhandled
    }
}
