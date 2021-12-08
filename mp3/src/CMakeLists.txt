add_executable(
    mp3
    main.cu
)

target_link_libraries(
    mp3
    PRIVATE
        libwb
)
