pub mod data;
pub mod window;
pub mod utils;

use gtk::prelude::*;
use gtk::{Application, ApplicationWindow};
use gtk::Orientation::Vertical;
use crate::window::BufferWindow;

const APP_ID: &str = "party.iroiro.juicemacs.gaudip";

fn main() -> glib::ExitCode {
    env_logger::init();
    glib::log_set_writer_func(glib_logger);

    let app = Application::builder().application_id(APP_ID).build();
    app.connect_activate(build_ui);
    app.run()
}

fn build_ui(app: &Application) {
    let layout = gtk::Box::new(Vertical, 0);
    let child = BufferWindow::new();
    layout.append(&child);

    let window = ApplicationWindow::builder()
        .application(app)
        .title("九叠")
        .child(&layout)
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
