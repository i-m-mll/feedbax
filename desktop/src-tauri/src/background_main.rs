fn main() {
    use feedbax_studio::service;
    if let Err(error) = service::run_background() {
        service::fail(error);
    }
}
