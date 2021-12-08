add_executable(
    mp5_1
    main.cu
)

target_link_libraries(
    mp5_1
    PRIVATE
        libwb
)
