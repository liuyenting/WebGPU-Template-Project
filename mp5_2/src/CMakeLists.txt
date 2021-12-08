add_executable(
    mp5_2
    main.cu
)

target_link_libraries(
    mp5_2
    PRIVATE
        libwb
)
