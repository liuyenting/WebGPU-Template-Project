add_executable(
    mp7
    main.cu
)

target_link_libraries(
    mp7
    PRIVATE
        libwb
)
