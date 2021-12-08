add_executable(
    mp6
    main.cu
)

target_link_libraries(
    mp6
    PRIVATE
        libwb
)
