from html.parser import HTMLParser
from pathlib import Path


PAGE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "design"
    / "visualizations"
    / "chengdu-greenhouse-experiment-flow.html"
)


class FlowPageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.tags = []
        self.text = []
        self.attributes = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append(tag)
        self.attributes.append(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])

    def handle_data(self, data):
        value = data.strip()
        if value:
            self.text.append(value)


def parse_page():
    parser = FlowPageParser()
    parser.feed(PAGE.read_text(encoding="utf-8"))
    return parser


def test_flow_page_contains_complete_scientific_workflow():
    parser = parse_page()
    required_sections = {
        "overview",
        "data",
        "models",
        "environment",
        "controllers",
        "protocol",
        "metrics",
    }
    assert required_sections <= parser.ids

    content = " ".join(parser.text)
    for controller in ("Baseline", "PID", "MPC", "PPO", "SAC"):
        assert controller in content
    for model in ("PINN", "Transformer", "ChengduPhysics", "Vanthoor"):
        assert model in content
    for metric in ("Reward", "MAE", "RMSE", "产量", "舒适区", "控制代价"):
        assert metric in content


def test_flow_page_states_evidence_limits_and_gates():
    parser = parse_page()
    content = " ".join(parser.text)
    assert "GO_OPEN_TEST" in content
    assert "NO_GO_PRODUCTION" in content
    assert "训练集" in content and "验证集" in content and "测试集" in content
    assert "测试集不能参与调参" in content
    assert "合成天气只用于训练" in content
    assert "没有成都目标温室真实采收序列" in content
    assert "加热和 CO2 固定为 0" in content


def test_flow_page_has_accessible_interactions():
    parser = parse_page()
    assert "details" in parser.tags
    assert "summary" in parser.tags
    assert "button" in parser.tags
    assert any(attrs.get("aria-label") for attrs in parser.attributes)
    assert any(attrs.get("data-metric-filter") for attrs in parser.attributes)

