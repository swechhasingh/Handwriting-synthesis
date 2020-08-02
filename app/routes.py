from flask import (
    request,
    render_template,
    flash,
    redirect,
    jsonify,
    make_response,
    url_for,
    session,
)
from app import flask_app
import os
import sys
import stat
import base64
import uuid
import numpy as np
from app.priming import generate_handwriting
from app.xml_parser import svg_xml_parser, path_to_stroke, path_string_to_stroke


# sys.path.append("../")
from utils import plot_stroke


@flask_app.route("/", methods=["GET"])
@flask_app.route("/about", methods=["GET"])
def index():
    return render_template("about.html", title="About")


@flask_app.route("/draw", methods=["GET"])
def draw():
    if "id" in session:
        id = session["id"]
        print("uuid: ", id)
    return render_template("draw.html", title="Write")


@flask_app.route("/upload_style", methods=["GET", "POST"])
def submit_style_data():
    data = request.get_json()
    path = data["path"]
    text = data["text"]
    if path == "":
        return jsonify(
            dict({"redirect": url_for("draw"), "message": "Please enter some style"})
        )

    id = str(uuid.uuid4())
    session["id"] = id
    tmp_dir = os.path.join(flask_app.root_path, "static", "uploads", id)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    os.chmod(tmp_dir, 0o777)
    print(tmp_dir)
    # user agent info
    user_agent = request.user_agent
    print(user_agent.string)
    print(user_agent.platform)
    phones = ["android", "iphone"]
    down_sample = True

    if user_agent.platform in phones:
        down_sample = False

    text_path = os.path.join(tmp_dir, "inpText.txt")
    print(text_path)
    with open(text_path, "w") as f:
        f.write(text)
    f.close()

    stroke = path_string_to_stroke(
        path, str_len=len(list(text)), down_sample=down_sample
    )
    save_path = os.path.join(tmp_dir, "style.npy")
    np.save(save_path, stroke, allow_pickle=True)
    print(save_path)

    # plot the sequence
    plot_stroke(stroke.astype(np.float32), os.path.join(tmp_dir, "original.png"))

    return jsonify(dict({"redirect": url_for("generate"), "message": ""}))


@flask_app.route("/generate", methods=["GET", "POST"])
def generate():
    default_style_path = os.path.join(
        flask_app.root_path, "static/uploads/default_style.npy"
    )
    org_img = base64.b64encode(
        open(
            os.path.join(flask_app.root_path, "static/uploads/default.png"), "rb"
        ).read()
    )
    org_src = "data:image/png;base64,{}".format(org_img.decode("ascii"))

    if request.method == "POST":
        text = request.form["text"]
        bias = float(request.form["bias"])
        style_option = request.form["styleOptions"]
        print("bias:{}, style_option:{}".format(bias, style_option))
        if text == "":
            message = "Please enter some text"
            return render_template(
                "generate.html",
                title="Generate",
                message=message,
                text="",
                org_src=org_src,
                samples="",
            )
        if style_option == "defaultStyle":
            style_path = default_style_path
            real_text = "copy monkey app"
            if not "id" in session:
                id = str(uuid.uuid4())
                session["id"] = id
            tmp_dir = os.path.join(
                flask_app.root_path, "static", "uploads", session["id"]
            )
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            os.chmod(tmp_dir, 0o777)
            print(tmp_dir)
        elif not "id" in session:
            return render_template(
                "generate.html",
                title="Generate",
                text="",
                message="Please go to Write and add some style.",
                org_src=org_src,
            )
        elif style_option == "yourStyle":
            id = session["id"]
            print("uuid", id)
            tmp_dir = os.path.join(
                flask_app.root_path, "static", "uploads", session["id"]
            )
            style_path = os.path.join(tmp_dir, "style.npy")
            if not os.path.exists(style_path):
                return render_template(
                    "generate.html",
                    title="Generate",
                    text="",
                    message="Please go to Write and add some style.",
                    org_src=org_src,
                )
            org_img_path = os.path.join(tmp_dir, "original.png")
            org_img = base64.b64encode(open(org_img_path, "rb").read())
            org_src = "data:image/png;base64,{}".format(org_img.decode("ascii"))

            text_path = os.path.join(tmp_dir, "inpText.txt")
            with open(text_path) as file:
                texts = file.read().splitlines()
            real_text = texts[0]

        save_path = os.path.join(tmp_dir)

        print(len(list(real_text)))
        print(real_text)
        n_samples = 5
        generate_handwriting(
            char_seq=text,
            real_text=real_text,
            style_path=style_path,
            save_path=save_path,
            app_path=flask_app.root_path,
            n_samples=n_samples,
            bias=bias,
        )

        gen_samp = []
        for i in range(n_samples):
            path = os.path.join(save_path, "gen_stroke_" + str(i) + ".png")
            encoded_image = base64.b64encode(open(path, "rb").read())
            src = "data:image/png;base64,{}".format(encoded_image.decode("ascii"))
            gen_samp.append(src)

        return render_template(
            "generate.html",
            title="Generate",
            text=text,
            bias=bias,
            org_src=org_src,
            samples=gen_samp,
        )

    elif request.method == "GET":
        if "id" in session:
            tmp_dir = os.path.join(
                flask_app.root_path, "static", "uploads", session["id"]
            )
            org_img_path = os.path.join(tmp_dir, "original.png")
            if os.path.exists(org_img_path):
                org_img = base64.b64encode(open(org_img_path, "rb").read())
                org_src = "data:image/png;base64,{}".format(org_img.decode("ascii"))

        return render_template(
            "generate.html",
            title="Generate",
            text="",
            org_src=org_src,
            samples="",
            message="",
        )
