add_executable(
    mp4
    main.cu
)

target_link_libraries(
    mp4
    PRIVATE
        libwb
)
